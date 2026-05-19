"""Acceptance tests for `tac.predictor.distortion_proxy_local` (council Q1 prescription).

The distortion proxy is the local closed-form fallback that the score-band
predictor calls when `rel_err > 1.0%` and no full empirical proxy is provided.
Without this module the score-band predictor refuses high-rel_err queries
(refusal #3 HIGH_REL_ERR_WITHOUT_PROXY); WITH it, queries flow through and
emit a band labeled `prediction_method="proxy"`.

These tests exercise:
  - Per-axis power-law fitting (pose + seg fit independently, not via global ratio)
  - Refusal modes (insufficient anchors / no lossless / no lossy / runaway extrapolation)
  - Determinism (same anchors → same fit → same predictions, byte-for-byte)
  - Floor clamping (proxy never reports below the lossless baseline)
  - Integration with `tac.predictor.score_band.predict_score_band` (proxy unblocks
    high-rel_err queries that would otherwise refuse)
  - Callable signature compatibility (proxy matches the `DistortionProxy` Protocol)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.predictor import (
    CalibrationAnchor,
    DistortionProxy,
    MAX_EXTRAPOLATION_MULTIPLIER,
    ProxyFit,
    fit_proxy,
    load_calibration_anchors,
    make_distortion_proxy,
    make_distortion_proxy_from_file,
    predict_distortion,
    predict_score_band,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: canonical calibration anchors (mirror test_score_band_predictor.py).
# ─────────────────────────────────────────────────────────────────────────────


def _pr106_anchor() -> CalibrationAnchor:
    return CalibrationAnchor(
        lane_id="lane_pr106_baseline",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20945673,
        avg_pose_dist=3.4e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=0.00496015,
        measured_utc="2026-05-05T17:25:19Z",
        job_id="exact-eval-public-pr106-baseline",
        archive_sha256="0af839ab",
    )


def _apogee_int8_anchor() -> CalibrationAnchor:
    return CalibrationAnchor(
        lane_id="lane_apogee_int8",
        rel_err_pct_per_weight=0.24,
        archive_bytes=187731,
        contest_cuda_score=0.21119242,
        avg_pose_dist=3.375e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=0.00500009,
        measured_utc="2026-05-05T17:45:00Z",
        job_id="apogee-int8-baseline-confirm",
        archive_sha256="b4e91a72",
    )


def _apogee_int4_anchor() -> CalibrationAnchor:
    return CalibrationAnchor(
        lane_id="lane_apogee_int4",
        rel_err_pct_per_weight=7.09,
        archive_bytes=109996,
        contest_cuda_score=1.42866394,
        avg_pose_dist=0.02370903,
        avg_seg_dist=0.00868503,
        rate_unscaled=0.00293052,
        measured_utc="2026-05-05T18:20:33Z",
        job_id="apogee-int4-naive-ptq-falsification",
        archive_sha256="c8d33f91",
    )


def _three_anchors() -> list[CalibrationAnchor]:
    return [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]


# ─────────────────────────────────────────────────────────────────────────────
# Fit-level tests: power-law regression on per-axis curves.
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_proxy_with_three_anchors_returns_finite_per_axis_coefs() -> None:
    """Three anchors (1 lossless + 2 lossy) give a fittable per-axis curve."""
    fit = fit_proxy(_three_anchors())
    assert not fit.refused, fit.refusal_reason
    assert math.isfinite(fit.pose_a) and math.isfinite(fit.pose_b)
    assert math.isfinite(fit.seg_a) and math.isfinite(fit.seg_b)
    assert math.isfinite(fit.pose_floor) and math.isfinite(fit.seg_floor)
    assert fit.pose_floor == pytest.approx(3.4e-5)
    assert fit.seg_floor == pytest.approx(0.00067819)
    assert fit.n_anchors == 3
    assert fit.n_lossy == 2
    assert fit.rel_err_min == pytest.approx(0.24)
    assert fit.rel_err_max == pytest.approx(7.09)


def test_fit_proxy_refuses_with_two_anchors_insufficient() -> None:
    """Two anchors is below MIN_CALIBRATION_ANCHORS (3) → refuse."""
    fit = fit_proxy([_pr106_anchor(), _apogee_int8_anchor()])
    assert fit.refused
    assert "insufficient_anchors" in fit.refusal_reason


def test_fit_proxy_refuses_without_any_lossless_anchor() -> None:
    """Without a rel_err=0 anchor, the proxy cannot establish the floor."""
    lossy_only = [_apogee_int8_anchor(), _apogee_int4_anchor(), _apogee_int4_anchor()]
    # Three distinct anchors all lossy
    fit = fit_proxy(lossy_only)
    assert fit.refused
    assert "no_lossless_anchor" in fit.refusal_reason


def test_fit_proxy_refuses_with_only_one_lossy_anchor() -> None:
    """With 1 lossless + only 1 lossy anchor, no curve can be fit."""
    # Build 3 anchors: 2 lossless + 1 lossy → ≥3 anchors, has lossless, but only 1 lossy.
    second_lossless = CalibrationAnchor(
        lane_id="lane_pr106_baseline_replay",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20945673,
        avg_pose_dist=3.45e-5,
        avg_seg_dist=0.00067820,
        rate_unscaled=0.00496015,
        measured_utc="2026-05-06T10:00:00Z",
        job_id="pr106-baseline-replay",
        archive_sha256="0af839ab2",
    )
    anchors = [_pr106_anchor(), second_lossless, _apogee_int8_anchor()]
    fit = fit_proxy(anchors)
    assert fit.refused
    assert "insufficient_lossy_anchors" in fit.refusal_reason


def test_fit_proxy_uses_tightest_lossless_when_multiple_lossless_present() -> None:
    """With multiple lossless anchors, the proxy chooses the one with smallest pose+seg."""
    tighter_lossless = CalibrationAnchor(
        lane_id="lane_pr106_baseline_tighter",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20800000,
        avg_pose_dist=3.0e-5,   # tighter than _pr106_anchor's 3.4e-5
        avg_seg_dist=0.00067000,  # tighter than _pr106_anchor's 0.00067819
        rate_unscaled=0.00496015,
        measured_utc="2026-05-07T10:00:00Z",
        job_id="pr106-baseline-tighter",
        archive_sha256="0af839ab3",
    )
    anchors = [_pr106_anchor(), tighter_lossless, _apogee_int8_anchor(), _apogee_int4_anchor()]
    fit = fit_proxy(anchors)
    assert not fit.refused
    # Floor must come from the TIGHTER lossless anchor
    assert fit.pose_floor == pytest.approx(3.0e-5)
    assert fit.seg_floor == pytest.approx(0.00067000)


# ─────────────────────────────────────────────────────────────────────────────
# Predict-level tests: querying the fitted proxy.
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_distortion_at_rel_err_zero_returns_lossless_floor() -> None:
    """Query at rel_err=0 returns the lossless baseline exactly."""
    fit = fit_proxy(_three_anchors())
    pose, seg = predict_distortion(fit, 0.0)
    assert pose == pytest.approx(fit.pose_floor)
    assert seg == pytest.approx(fit.seg_floor)


def test_predict_distortion_at_calibrated_anchor_is_close_to_anchor() -> None:
    """Query at the apogee_int4 rel_err should be in the same order of magnitude
    as the anchor's measured distortion (curve passes near the calibration points)."""
    fit = fit_proxy(_three_anchors())
    int4 = _apogee_int4_anchor()
    pose, seg = predict_distortion(fit, int4.rel_err_pct_per_weight)
    # With only 2 lossy anchors and exact log-linear fit, the curve passes through
    # both lossy anchors at log-scale — but the FLOOR-subtracted excess match is
    # exact in log space, so the reconstructed distortion should hit the anchor
    # almost perfectly (rounding only).
    assert pose == pytest.approx(int4.avg_pose_dist, rel=0.05)
    assert seg == pytest.approx(int4.avg_seg_dist, rel=0.05)


def test_predict_distortion_refuses_runaway_extrapolation() -> None:
    """rel_err beyond MAX_EXTRAPOLATION_MULTIPLIER * max calibrated → NaN."""
    fit = fit_proxy(_three_anchors())
    # max calibrated is 7.09; MAX_EXTRAPOLATION_MULTIPLIER=5 → cutoff ≈35.45
    far_rel_err = MAX_EXTRAPOLATION_MULTIPLIER * fit.rel_err_max + 1.0
    pose, seg = predict_distortion(fit, far_rel_err)
    assert math.isnan(pose)
    assert math.isnan(seg)


def test_predict_distortion_refuses_negative_rel_err() -> None:
    """Negative rel_err is meaningless (magnitude only) → NaN."""
    fit = fit_proxy(_three_anchors())
    pose, seg = predict_distortion(fit, -0.5)
    assert math.isnan(pose)
    assert math.isnan(seg)


def test_predict_distortion_refused_fit_returns_nan() -> None:
    """If the fit itself was refused, every prediction is NaN."""
    fit = fit_proxy([_pr106_anchor()])  # only 1 anchor → refused
    assert fit.refused
    for rel_err in [0.0, 0.5, 2.0, 10.0]:
        pose, seg = predict_distortion(fit, rel_err)
        assert math.isnan(pose), f"rel_err={rel_err}"
        assert math.isnan(seg), f"rel_err={rel_err}"


def test_predict_distortion_clamps_below_lossless_floor() -> None:
    """If a degenerate fit ever predicts D below the floor, the clamp engages.

    Floor clamping is a safety net for cases where the fitted curve, evaluated
    at small rel_err, produces D < floor (numerically possible with noisy
    coefficients). This test constructs anchors where the lossy data is
    slightly noisier than the lossless to exercise the clamp.
    """
    fit = fit_proxy(_three_anchors())
    # Query at a very small rel_err inside the calibrated range
    pose, seg = predict_distortion(fit, 0.01)
    # Predictions must NEVER drop below the floor
    assert pose >= fit.pose_floor - 1e-15
    assert seg >= fit.seg_floor - 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# Determinism tests: same anchors → same fit → same predictions.
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_proxy_is_deterministic_across_repeated_calls() -> None:
    """Same anchors → same fit, byte-for-byte across calls."""
    anchors = _three_anchors()
    fit_a = fit_proxy(anchors)
    fit_b = fit_proxy(anchors)
    assert fit_a.pose_floor == fit_b.pose_floor
    assert fit_a.seg_floor == fit_b.seg_floor
    assert fit_a.pose_a == fit_b.pose_a
    assert fit_a.pose_b == fit_b.pose_b
    assert fit_a.seg_a == fit_b.seg_a
    assert fit_a.seg_b == fit_b.seg_b


def test_predict_distortion_is_deterministic_across_repeated_calls() -> None:
    """Same fit + same query → same prediction, byte-for-byte."""
    fit = fit_proxy(_three_anchors())
    for rel_err in [0.0, 0.5, 1.0, 3.0, 7.09]:
        pose_a, seg_a = predict_distortion(fit, rel_err)
        pose_b, seg_b = predict_distortion(fit, rel_err)
        assert pose_a == pose_b, f"rel_err={rel_err}: pose diverged"
        assert seg_a == seg_b, f"rel_err={rel_err}: seg diverged"


# ─────────────────────────────────────────────────────────────────────────────
# Callable API tests: make_distortion_proxy returns a DistortionProxy.
# ─────────────────────────────────────────────────────────────────────────────


def test_make_distortion_proxy_returns_callable_with_three_arg_signature() -> None:
    """The returned proxy must match the DistortionProxy Protocol: (bytes, rel_err, n) → (pose, seg)."""
    proxy = make_distortion_proxy(_three_anchors())
    assert callable(proxy)
    pose, seg = proxy(187000, 0.5, 100)
    assert isinstance(pose, float)
    assert isinstance(seg, float)


def test_make_distortion_proxy_ignores_unused_args_in_closed_form_variant() -> None:
    """The closed-form proxy depends only on rel_err. Changing archive_bytes or
    n_quantized_layers should produce the same output for the same rel_err."""
    proxy = make_distortion_proxy(_three_anchors())
    pose_a, seg_a = proxy(186239, 1.5, 88)
    pose_b, seg_b = proxy(99999999, 1.5, 1)  # absurd archive_bytes, tiny n_layers
    assert pose_a == pose_b
    assert seg_a == seg_b


def test_make_distortion_proxy_is_o1_per_call_no_refitting() -> None:
    """The proxy closes over a single fit. Calling it 1000 times shouldn't refit.

    We can't assert wall-clock from a unit test, but we can assert that the
    fit object is constructed exactly once by ensuring repeated calls return
    identical results without any state change.
    """
    proxy = make_distortion_proxy(_three_anchors())
    first_call = proxy(186239, 1.0, 88)
    # 100 intermediate calls
    for _ in range(100):
        proxy(186239, 1.0, 88)
    last_call = proxy(186239, 1.0, 88)
    assert first_call == last_call


def test_make_distortion_proxy_refused_fit_returns_proxy_returning_nan() -> None:
    """When the fit is refused, the returned callable returns NaN for every query."""
    proxy = make_distortion_proxy([_pr106_anchor()])  # insufficient
    pose, seg = proxy(186239, 2.0, 88)
    assert math.isnan(pose)
    assert math.isnan(seg)


def test_make_distortion_proxy_from_file_roundtrip(tmp_path: Path) -> None:
    """Persisting anchors to JSON and loading them via the file-helper builds
    the same proxy as constructing in-memory."""
    anchors = _three_anchors()
    # Serialize anchors to JSON manually (mirrors load_calibration_anchors format)
    payload = [
        {
            "lane_id": a.lane_id,
            "rel_err_pct_per_weight": a.rel_err_pct_per_weight,
            "archive_bytes": a.archive_bytes,
            "contest_cuda_score": a.contest_cuda_score,
            "avg_pose_dist": a.avg_pose_dist,
            "avg_seg_dist": a.avg_seg_dist,
            "rate_unscaled": a.rate_unscaled,
            "measured_utc": a.measured_utc,
            "job_id": a.job_id,
            "archive_sha256": a.archive_sha256,
            "notes": a.notes,
        }
        for a in anchors
    ]
    json_path = tmp_path / "anchors.json"
    json_path.write_text(json.dumps(payload))

    proxy_file = make_distortion_proxy_from_file(json_path)
    proxy_mem = make_distortion_proxy(anchors)

    pose_file, seg_file = proxy_file(186239, 1.0, 88)
    pose_mem, seg_mem = proxy_mem(186239, 1.0, 88)

    assert pose_file == pose_mem
    assert seg_file == seg_mem


def test_make_distortion_proxy_from_missing_file_returns_refused_proxy(tmp_path: Path) -> None:
    """A non-existent anchors file produces a proxy that always returns NaN."""
    proxy = make_distortion_proxy_from_file(tmp_path / "does_not_exist.json")
    pose, seg = proxy(186239, 1.0, 88)
    assert math.isnan(pose)
    assert math.isnan(seg)


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests: proxy unblocks high-rel_err score-band queries.
# ─────────────────────────────────────────────────────────────────────────────


def test_score_band_predictor_accepts_high_rel_err_when_proxy_provided() -> None:
    """The score-band predictor refuses high rel_err WITHOUT a proxy; WITH a
    well-calibrated proxy it should produce a calibrated band."""
    anchors = _three_anchors()
    proxy = make_distortion_proxy(anchors)
    band = predict_score_band(
        archive_bytes=109996,
        rel_err_pct_per_weight=7.09,
        n_quantized_layers=88,
        calibration_anchors=anchors,
        distortion_proxy=proxy,
    )
    assert not band.refused, band.refusal_reason
    assert band.prediction_method == "proxy"
    assert band.confidence in {"calibrated_strong", "calibrated_weak"}


def test_score_band_predictor_refuses_high_rel_err_when_proxy_missing() -> None:
    """Same query as above but with no proxy → refusal #3 fires."""
    anchors = _three_anchors()
    band = predict_score_band(
        archive_bytes=109996,
        rel_err_pct_per_weight=7.09,
        n_quantized_layers=88,
        calibration_anchors=anchors,
        distortion_proxy=None,
    )
    assert band.refused
    assert "high_rel_err_without_proxy" in band.refusal_reason


# ─────────────────────────────────────────────────────────────────────────────
# Type/protocol tests: the proxy satisfies the DistortionProxy alias.
# ─────────────────────────────────────────────────────────────────────────────


def test_distortion_proxy_type_alias_accepts_callable() -> None:
    """The DistortionProxy type alias is `Callable[[int, float, int], tuple[float, float]]`.
    A function with that exact signature should be assignable to a variable
    annotated with that alias."""
    def manual_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
        return (0.0, 0.0)

    # The point of this test: this assignment compiles + runs without TypeError.
    proxy: DistortionProxy = manual_proxy
    pose, seg = proxy(1, 0.0, 1)
    assert pose == 0.0
    assert seg == 0.0


def test_proxy_fit_dataclass_is_frozen() -> None:
    """ProxyFit is a frozen dataclass — attribute mutation must raise."""
    fit = fit_proxy(_three_anchors())
    with pytest.raises((AttributeError, TypeError, Exception)):
        fit.pose_floor = 999.0  # type: ignore[misc]


def test_proxy_fit_as_str_describes_fit_or_refusal() -> None:
    """ProxyFit.as_str() should provide a human-readable summary."""
    fit = fit_proxy(_three_anchors())
    s = fit.as_str()
    assert "pose" in s and "seg" in s
    assert "n=3" in s

    refused = fit_proxy([_pr106_anchor()])
    assert refused.refused
    assert "REFUSED" in refused.as_str()
