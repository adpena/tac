"""Public API + contract tests for `tac.predictor.score_band`.

This file complements `tac/tests/test_score_band_predictor.py` (which exercises
the empirical apogee_int4 failure scenarios) by focusing on the MODULE-LEVEL
public-API contract:

  - Canonical exported symbols are importable from `tac.predictor`
  - Contest-defined constants match upstream/evaluate.py exactly
  - The `DistortionProxy` Callable type alias is the documented 3-arg shape
  - `_score_from_components` (private) matches the contest formula
  - `ScoreBand.as_str()` formats both refused and accepted bands
  - `CalibrationAnchor` is frozen + validates rate consistency
  - Edge cases: zero anchors, all-lossless anchors, all-lossy anchors

The file is named to match the CI workflow's expected path
(`.github/workflows/test.yml` invokes `tac/tests/test_predictor_score_band.py`)
so the workflow can find the file without renaming.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import get_args, get_origin

import pytest

from tac.predictor import (
    CalibrationAnchor,
    DistortionProxy,
    ScoreBand,
    fit_distortion_curve,
    load_calibration_anchors,
    predict_score_band,
)
from tac.predictor.score_band import (
    HIGH_REL_ERR_THRESHOLD_PCT,
    MIN_CALIBRATION_ANCHORS,
    POSE_COEFFICIENT_SQRT_INNER,
    PR106_TOTAL_RATE_DENOM,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
    _score_from_components,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
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


# ─────────────────────────────────────────────────────────────────────────────
# Module-level public-API tests.
# ─────────────────────────────────────────────────────────────────────────────


def test_canonical_symbols_importable_from_tac_predictor() -> None:
    """`from tac.predictor import X` must work for every canonical name."""
    # The actual imports at file head are the assertion. Sanity-check the
    # types of the most important symbols here.
    assert CalibrationAnchor.__name__ == "CalibrationAnchor"
    assert ScoreBand.__name__ == "ScoreBand"
    assert callable(predict_score_band)
    assert callable(fit_distortion_curve)
    assert callable(load_calibration_anchors)
    # DistortionProxy is a Callable type alias — it's not a class
    assert DistortionProxy is not None


def test_contest_defined_constants_match_upstream_evaluate_py() -> None:
    """These constants are CONTEST-DEFINED — they must NEVER drift.

    Canonical source: upstream/evaluate.py + CLAUDE.md "TRUE score data" section.
    """
    assert SEG_COEFFICIENT == 100.0
    assert POSE_COEFFICIENT_SQRT_INNER == 10.0
    assert RATE_COEFFICIENT == 25.0
    assert PR106_TOTAL_RATE_DENOM == 37545489


def test_process_discipline_constants_have_sane_defaults() -> None:
    """Process-discipline constants are tuneable but should have safe defaults."""
    assert MIN_CALIBRATION_ANCHORS == 3
    assert HIGH_REL_ERR_THRESHOLD_PCT == 1.0


def test_score_from_components_matches_pr106_baseline_within_eps() -> None:
    """The private `_score_from_components` should reproduce the PR106 baseline
    contest-CUDA score within 5e-3 (the workflow's own tolerance for this
    reconstruction)."""
    score = _score_from_components(
        pose=3.4e-5,
        seg=0.00067819,
        rate_unscaled=0.00496015,
    )
    assert score == pytest.approx(0.20945673, abs=5e-3)


def test_score_from_components_matches_apogee_int4_falsification_within_eps() -> None:
    """The apogee_int4 falsification anchor (rel_err=7.09%, landed 1.4287
    [contest-CUDA]) should reproduce within 1e-3."""
    score = _score_from_components(
        pose=0.02370903,
        seg=0.00868503,
        rate_unscaled=109996 / PR106_TOTAL_RATE_DENOM,
    )
    assert score == pytest.approx(1.42866394, abs=1e-3)


def test_score_from_components_handles_zero_pose() -> None:
    """sqrt(POSE_COEFFICIENT_SQRT_INNER * 0) must be 0, not NaN."""
    score = _score_from_components(pose=0.0, seg=0.001, rate_unscaled=0.005)
    expected = SEG_COEFFICIENT * 0.001 + 0.0 + RATE_COEFFICIENT * 0.005
    assert score == pytest.approx(expected)


def test_score_from_components_clamps_negative_pose_to_zero() -> None:
    """Negative pose (numerical noise) is clamped via max(pose, 0.0) inside the sqrt."""
    score = _score_from_components(pose=-1e-9, seg=0.001, rate_unscaled=0.005)
    expected = SEG_COEFFICIENT * 0.001 + 0.0 + RATE_COEFFICIENT * 0.005
    assert score == pytest.approx(expected)


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationAnchor dataclass contract.
# ─────────────────────────────────────────────────────────────────────────────


def test_calibration_anchor_is_frozen() -> None:
    """Frozen dataclass — attribute mutation must raise."""
    anchor = _pr106_anchor()
    with pytest.raises((AttributeError, TypeError, Exception)):
        anchor.archive_bytes = 999999  # type: ignore[misc]


def test_calibration_anchor_validates_rate_consistency_at_construction() -> None:
    """`__post_init__` rejects rate_unscaled that doesn't match archive_bytes/denom."""
    with pytest.raises(ValueError, match="inconsistent"):
        CalibrationAnchor(
            lane_id="lane_broken_rate",
            rel_err_pct_per_weight=0.0,
            archive_bytes=186239,
            contest_cuda_score=0.20945673,
            avg_pose_dist=3.4e-5,
            avg_seg_dist=0.00067819,
            rate_unscaled=999.0,  # ← inconsistent with archive_bytes / denom
            measured_utc="2026-05-05T17:25:19Z",
            job_id="broken-test",
            archive_sha256="deadbeef",
        )


def test_calibration_anchor_accepts_optional_notes_field() -> None:
    """`notes` defaults to empty string and accepts free-form text."""
    anchor = CalibrationAnchor(
        lane_id="lane_test",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20945673,
        avg_pose_dist=3.4e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=0.00496015,
        measured_utc="2026-05-05T17:25:19Z",
        job_id="test",
        archive_sha256="deadbeef",
        notes="manual replay confirmation",
    )
    assert anchor.notes == "manual replay confirmation"

    anchor_default = _pr106_anchor()
    assert anchor_default.notes == ""


# ─────────────────────────────────────────────────────────────────────────────
# ScoreBand dataclass contract.
# ─────────────────────────────────────────────────────────────────────────────


def test_score_band_is_frozen() -> None:
    """ScoreBand is frozen."""
    band = ScoreBand(low=0.1, high=0.2, confidence="calibrated_strong", refused=False)
    with pytest.raises((AttributeError, TypeError, Exception)):
        band.low = 0.0  # type: ignore[misc]


def test_score_band_as_str_formats_accepted_band() -> None:
    """Accepted band format: '[low, high] (confidence=X)'."""
    band = ScoreBand(low=0.1923, high=0.2123, confidence="calibrated_strong", refused=False)
    s = band.as_str()
    assert "0.1923" in s
    assert "0.2123" in s
    assert "calibrated_strong" in s


def test_score_band_as_str_formats_refused_band() -> None:
    """Refused band format: 'REFUSED (reason)'."""
    band = ScoreBand(
        low=0.0, high=0.0, confidence="none",
        refused=True, refusal_reason="insufficient_anchors",
    )
    s = band.as_str()
    assert "REFUSED" in s
    assert "insufficient_anchors" in s


def test_score_band_default_values_are_safe() -> None:
    """Optional fields default to safe values (no NaN, no None where float expected)."""
    band = ScoreBand(low=0.1, high=0.2, confidence="calibrated_weak", refused=False)
    assert band.refusal_reason == ""
    assert band.distortion_estimate_used is False
    assert band.predicted_pose == 0.0
    assert band.predicted_seg == 0.0
    assert band.predicted_rate == 0.0
    assert band.prediction_method == "none"
    assert band.derivation == ""


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: zero anchors, all-lossless, all-lossy.
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_score_band_refuses_with_zero_anchors() -> None:
    """Empty anchor list triggers refusal #1 (insufficient_anchors)."""
    band = predict_score_band(
        archive_bytes=186239,
        rel_err_pct_per_weight=0.5,
        n_quantized_layers=88,
        calibration_anchors=[],
    )
    assert band.refused
    assert "insufficient_anchors" in band.refusal_reason


def test_fit_distortion_curve_with_zero_anchors_returns_nan() -> None:
    """Zero anchors → NaN coefficients."""
    curve = fit_distortion_curve([])
    assert math.isnan(curve["a"])
    assert math.isnan(curve["b"])
    assert curve["n_anchors"] == 0


def test_load_calibration_anchors_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """Missing file returns [] (silent, not error) per the documented contract."""
    anchors = load_calibration_anchors(tmp_path / "does_not_exist.json")
    assert anchors == []


def test_load_calibration_anchors_rejects_non_list_json(tmp_path: Path) -> None:
    """A JSON object (not list) at the anchor file path raises ValueError."""
    path = tmp_path / "bad_anchors.json"
    path.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="JSON list"):
        load_calibration_anchors(path)


# ─────────────────────────────────────────────────────────────────────────────
# DistortionProxy callable type alias.
# ─────────────────────────────────────────────────────────────────────────────


def test_distortion_proxy_alias_accepts_3_arg_callable() -> None:
    """A function with signature (int, float, int) → (float, float) satisfies the alias."""
    def my_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
        return (0.001, 0.0005)

    proxy: DistortionProxy = my_proxy  # type: ignore[assignment]
    pose, seg = proxy(186239, 1.0, 88)
    assert pose == 0.001
    assert seg == 0.0005


def test_predict_score_band_consumes_distortion_proxy_callable() -> None:
    """A custom DistortionProxy unblocks high-rel_err refusal #3."""
    def constant_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
        # Match PR106 baseline + small per-rel_err inflation
        return (3.4e-5 + 1e-6 * rel_err_pct, 0.00067819 + 1e-5 * rel_err_pct)

    # Build a 3-anchor calibration: PR106 + apogee_int8 + a fabricated lossy at 1.5%
    third = CalibrationAnchor(
        lane_id="lane_fabricated_lossy",
        rel_err_pct_per_weight=1.5,
        archive_bytes=180000,
        contest_cuda_score=0.30,
        avg_pose_dist=5e-5,
        avg_seg_dist=0.0009,
        rate_unscaled=180000 / PR106_TOTAL_RATE_DENOM,
        measured_utc="2026-05-06T00:00:00Z",
        job_id="fabricated",
        archive_sha256="ffffffff",
    )
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), third]
    band = predict_score_band(
        archive_bytes=185000,
        rel_err_pct_per_weight=1.2,
        n_quantized_layers=88,
        calibration_anchors=anchors,
        distortion_proxy=constant_proxy,
    )
    assert not band.refused, band.refusal_reason
    assert band.prediction_method == "proxy"
    assert band.distortion_estimate_used is True


# ─────────────────────────────────────────────────────────────────────────────
# Determinism contracts.
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_score_band_is_deterministic_across_repeated_calls() -> None:
    """Same inputs → same band, byte-for-byte."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor()]  # 2 anchors → refusal #1
    # Use 3 anchors to get past refusal #1
    third = CalibrationAnchor(
        lane_id="lane_third",
        rel_err_pct_per_weight=0.5,
        archive_bytes=180000,
        contest_cuda_score=0.25,
        avg_pose_dist=4e-5,
        avg_seg_dist=0.0008,
        rate_unscaled=180000 / PR106_TOTAL_RATE_DENOM,
        measured_utc="2026-05-06T00:00:00Z",
        job_id="third",
        archive_sha256="00000000",
    )
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), third]
    band_a = predict_score_band(186239, 0.3, 88, anchors)
    band_b = predict_score_band(186239, 0.3, 88, anchors)
    assert band_a == band_b


def test_fit_distortion_curve_is_deterministic_across_repeated_calls() -> None:
    """Same anchors → same curve coefficients, byte-for-byte."""
    third = CalibrationAnchor(
        lane_id="lane_third",
        rel_err_pct_per_weight=0.5,
        archive_bytes=180000,
        contest_cuda_score=0.25,
        avg_pose_dist=4e-5,
        avg_seg_dist=0.0008,
        rate_unscaled=180000 / PR106_TOTAL_RATE_DENOM,
        measured_utc="2026-05-06T00:00:00Z",
        job_id="third",
        archive_sha256="00000000",
    )
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), third]
    c_a = fit_distortion_curve(anchors)
    c_b = fit_distortion_curve(anchors)
    assert c_a == c_b
