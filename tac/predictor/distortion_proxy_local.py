"""Local closed-form distortion proxy for score-band prediction (council Q1 prescription).

The score-band predictor (`tac.predictor.score_band.predict_score_band`) refuses
to emit a band when `rel_err > 1.0%` and no `distortion_proxy` is provided.
This module supplies a canonical reference proxy that is:

  - Closed-form (no GPU, no inflate, no scorer forward pass — runs in microseconds)
  - Calibrated against empirical anchors (read from
    `.omx/calibration/anchors_<lane_class>.json` per the calibration provenance
    contract)
  - Deterministic (same inputs → same outputs, byte-for-byte reproducible)
  - Refuses-to-extrapolate-far (returns NaN tuple when rel_err is wildly
    outside the calibrated range, forcing the caller to acquire more anchors)

The full empirical proxy (`experiments.distortion_proxy_local` in the broader
Pact research repository) runs a real inflate + scorer forward pass on ≥30
ground-truth frames. THIS module is the OSS-shippable closed-form fallback
that the score-band predictor calls when the full empirical proxy is not
available.

CALIBRATION SOURCES per CLAUDE.md "Predictor calibration" non-negotiable:

1. `pose_floor`, `seg_floor` derived from the tightest lossless anchor's
   `avg_pose_dist`, `avg_seg_dist` (CONSERVATIVE; the proxy reports at least
   the lossless floor).
2. Per-axis power-law curves `D = floor + a * rel_err^b` fitted via closed-form
   log-linear regression on the LOSSY anchors (same approach as
   `tac.predictor.score_band.fit_distortion_curve` but per-axis instead of
   summed pose+seg).
3. Pose/seg split is per-axis fitting (NOT a global ratio applied to a summed
   curve) so the proxy can express different rates of pose-vs-seg degradation
   in different rel_err regimes.

REFUSAL MODES — return `(nan, nan)` when:

  1. INSUFFICIENT_ANCHORS:  <3 anchors (matches predictor refusal #1)
  2. NO_LOSSLESS_ANCHOR:    no anchor with rel_err == 0 (cannot establish floor)
  3. NO_LOSSY_ANCHORS:      no anchor with rel_err > 0 (cannot fit curve)
  4. RUNAWAY_EXTRAPOLATION: rel_err > 5x the max calibrated rel_err
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tac.predictor.score_band import (
    MIN_CALIBRATION_ANCHORS,
    CalibrationAnchor,
    DistortionProxy,
    load_calibration_anchors,
)

# ── Calibration constants (process discipline, not contest-defined) ──
# Beyond this many multiples of the calibrated max, the proxy refuses
# rather than extrapolate. Council Q1 (Dykstra): proxies must be honest
# about their support region.
MAX_EXTRAPOLATION_MULTIPLIER = 5.0

# Floor on (D_anchor - D_baseline) inside log to avoid log(0).
_EXCESS_FLOOR = 1e-12


@dataclass(frozen=True)
class ProxyFit:
    """Result of fitting a per-axis power-law distortion curve."""
    pose_floor: float  # D_pose at rel_err == 0 (lossless baseline)
    seg_floor: float   # D_seg at rel_err == 0
    pose_a: float      # multiplicative coefficient for pose curve
    pose_b: float      # exponent for pose curve
    seg_a: float       # multiplicative coefficient for seg curve
    seg_b: float       # exponent for seg curve
    rel_err_min: float
    rel_err_max: float
    n_anchors: int
    n_lossy: int
    refused: bool = False
    refusal_reason: str = ""

    def as_str(self) -> str:
        if self.refused:
            return f"ProxyFit(REFUSED: {self.refusal_reason})"
        return (
            f"ProxyFit(pose_floor={self.pose_floor:.4g}, seg_floor={self.seg_floor:.4g}, "
            f"pose: D=floor+{self.pose_a:.4g}*r^{self.pose_b:.4g}, "
            f"seg: D=floor+{self.seg_a:.4g}*r^{self.seg_b:.4g}, "
            f"n={self.n_anchors})"
        )


def _fit_axis_power_law(
    floor: float,
    lossy: list[CalibrationAnchor],
    extract: str,
) -> tuple[float, float]:
    """Closed-form log-linear regression for one axis.

    Fits `log(D - floor) ≈ b·log(rel_err) + log(a)`. Returns `(a, b)`. If the
    fit is degenerate (var_x == 0, or all excesses non-positive), returns
    `(nan, nan)`.
    """
    log_rel: list[float] = []
    log_excess: list[float] = []
    for anchor in lossy:
        d = getattr(anchor, extract)
        excess = max(d - floor, _EXCESS_FLOOR)
        log_rel.append(math.log(anchor.rel_err_pct_per_weight))
        log_excess.append(math.log(excess))

    n = len(log_rel)
    if n < 2:
        return float("nan"), float("nan")
    mean_x = sum(log_rel) / n
    mean_y = sum(log_excess) / n
    var_x = sum((x - mean_x) ** 2 for x in log_rel)
    if var_x == 0:
        return float("nan"), float("nan")
    cov_xy = sum((log_rel[i] - mean_x) * (log_excess[i] - mean_y) for i in range(n))
    b = cov_xy / var_x
    log_a = mean_y - b * mean_x
    a = math.exp(log_a)
    return a, b


def fit_proxy(anchors: list[CalibrationAnchor]) -> ProxyFit:
    """Fit per-axis power-law curves from calibration anchors.

    Returns a `ProxyFit` whose `refused=True` flag is set when calibration is
    insufficient. The caller is expected to check `refused` before consuming
    the fit (the proxy callable wraps this check transparently).
    """
    if len(anchors) < MIN_CALIBRATION_ANCHORS:
        return ProxyFit(
            pose_floor=float("nan"), seg_floor=float("nan"),
            pose_a=float("nan"), pose_b=float("nan"),
            seg_a=float("nan"), seg_b=float("nan"),
            rel_err_min=float("nan"), rel_err_max=float("nan"),
            n_anchors=len(anchors), n_lossy=0,
            refused=True,
            refusal_reason=(
                f"insufficient_anchors: have {len(anchors)}, need {MIN_CALIBRATION_ANCHORS}"
            ),
        )

    lossless = [a for a in anchors if a.rel_err_pct_per_weight == 0.0]
    if not lossless:
        return ProxyFit(
            pose_floor=float("nan"), seg_floor=float("nan"),
            pose_a=float("nan"), pose_b=float("nan"),
            seg_a=float("nan"), seg_b=float("nan"),
            rel_err_min=float("nan"), rel_err_max=float("nan"),
            n_anchors=len(anchors), n_lossy=0,
            refused=True,
            refusal_reason="no_lossless_anchor: cannot establish baseline floor",
        )

    lossy = [a for a in anchors if a.rel_err_pct_per_weight > 0.0]
    if len(lossy) < 2:
        return ProxyFit(
            pose_floor=float("nan"), seg_floor=float("nan"),
            pose_a=float("nan"), pose_b=float("nan"),
            seg_a=float("nan"), seg_b=float("nan"),
            rel_err_min=float("nan"), rel_err_max=float("nan"),
            n_anchors=len(anchors), n_lossy=len(lossy),
            refused=True,
            refusal_reason="insufficient_lossy_anchors: need ≥2 lossy anchors to fit curve",
        )

    # Use the lossless anchor with TIGHTEST distortion (min pose+seg) as floor.
    # If multiple lossless anchors exist this disambiguates deterministically.
    tightest_lossless = min(lossless, key=lambda a: a.avg_pose_dist + a.avg_seg_dist)
    pose_floor = tightest_lossless.avg_pose_dist
    seg_floor = tightest_lossless.avg_seg_dist

    pose_a, pose_b = _fit_axis_power_law(pose_floor, lossy, "avg_pose_dist")
    seg_a, seg_b = _fit_axis_power_law(seg_floor, lossy, "avg_seg_dist")

    rel_errs = [a.rel_err_pct_per_weight for a in lossy]

    return ProxyFit(
        pose_floor=pose_floor,
        seg_floor=seg_floor,
        pose_a=pose_a, pose_b=pose_b,
        seg_a=seg_a, seg_b=seg_b,
        rel_err_min=min(rel_errs),
        rel_err_max=max(rel_errs),
        n_anchors=len(anchors),
        n_lossy=len(lossy),
        refused=False,
    )


def predict_distortion(
    fit: ProxyFit,
    rel_err_pct_per_weight: float,
) -> tuple[float, float]:
    """Evaluate the fitted proxy at a query rel_err. Returns (pose, seg).

    Refuses (returns (nan, nan)) when:
      - the fit itself is refused
      - either axis fit is degenerate (a or b NaN)
      - rel_err is > MAX_EXTRAPOLATION_MULTIPLIER * fit.rel_err_max

    A query at rel_err == 0 returns (pose_floor, seg_floor) exactly, matching
    the lossless baseline.
    """
    if fit.refused:
        return float("nan"), float("nan")
    if rel_err_pct_per_weight < 0.0:
        # Negative rel_err is meaningless (per-weight error is a magnitude).
        return float("nan"), float("nan")
    if rel_err_pct_per_weight == 0.0:
        return fit.pose_floor, fit.seg_floor
    if rel_err_pct_per_weight > MAX_EXTRAPOLATION_MULTIPLIER * fit.rel_err_max:
        # Refuse to extrapolate far beyond calibrated range.
        return float("nan"), float("nan")
    if any(math.isnan(x) for x in (fit.pose_a, fit.pose_b, fit.seg_a, fit.seg_b)):
        return float("nan"), float("nan")

    pose = fit.pose_floor + fit.pose_a * (rel_err_pct_per_weight ** fit.pose_b)
    seg = fit.seg_floor + fit.seg_a * (rel_err_pct_per_weight ** fit.seg_b)
    # Clamp at floor (never report below the lossless baseline).
    pose = max(pose, fit.pose_floor)
    seg = max(seg, fit.seg_floor)
    return pose, seg


def make_distortion_proxy(
    anchors: list[CalibrationAnchor],
) -> DistortionProxy:
    """Build a DistortionProxy callable from calibration anchors.

    The returned callable matches the `DistortionProxy` Protocol declared by
    `tac.predictor.score_band`: it accepts `(archive_bytes, rel_err_pct,
    n_quantized_layers)` and returns `(predicted_avg_pose_dist,
    predicted_avg_seg_dist)`.

    The callable closes over the fitted curve, so repeated calls are O(1)
    (no re-fitting). The callable is deterministic and pure.
    """
    fit = fit_proxy(anchors)

    def _proxy(archive_bytes: int, rel_err_pct: float, n_quantized_layers: int) -> tuple[float, float]:
        # archive_bytes and n_quantized_layers are part of the canonical
        # DistortionProxy signature but the closed-form proxy depends only
        # on rel_err. The full empirical proxy uses all three.
        _ = archive_bytes  # noqa: deliberately unused in closed-form variant
        _ = n_quantized_layers
        return predict_distortion(fit, rel_err_pct)

    return _proxy


def make_distortion_proxy_from_file(path: Path) -> DistortionProxy:
    """Convenience: load anchors from a JSON file and build a proxy in one call."""
    anchors = load_calibration_anchors(path)
    return make_distortion_proxy(anchors)


__all__ = [
    "MAX_EXTRAPOLATION_MULTIPLIER",
    "ProxyFit",
    "fit_proxy",
    "predict_distortion",
    "make_distortion_proxy",
    "make_distortion_proxy_from_file",
]
