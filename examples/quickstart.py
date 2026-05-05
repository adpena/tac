"""tac quickstart — meta-Lagrangian search + closed-loop feedback on a TOY problem.

Run::

    python examples/quickstart.py

This script needs no GPU, no comma-challenge archive, and no checked-in data.
It exercises the four primitives that `tac` exposes for orchestrating an
extreme automated codec search:

  1. Build 3 synthetic calibration *anchors* (rel_err -> contest-CUDA score).
  2. Rank 5 synthetic candidates with the Boyd-style Lagrangian search.
  3. Show how the predictor's score band *widens / refuses* when the
     distortion proxy degenerates outside the calibration regime.
  4. Sketch the parallel-dispatch + harvest-and-reseed actuator pattern
     (mocked — no real GPU is touched).

The toy problem mirrors the post-contest closed-loop architecture documented
in `docs/paper/07_discussion.md` §7.8 in the parent comma-lab repository:
the May 4 race-window failure mode was that we built the *ranker* without the
*actuator*. This quickstart runs both ends so the contract is visible.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from tac.optimizer import (
    LagrangianConstraints,
    MetaLagrangianSearch,
    contest_score,
)
from tac.predictor import (
    CalibrationAnchor,
    predict_score_band,
)


# ── 1. Synthetic anchors ─────────────────────────────────────────────────────
#
# Three anchors define a curve in (rel_err, distortion) space. The predictor
# refuses with `insufficient_anchors` if you hand it fewer than 3.
#
# Anchor values are dimensionless toy numbers: a "lossless" baseline (rel_err 0,
# matches the PR106 frontier in shape), one mid-point, one aggressive lossy
# anchor where distortion is meaningfully worse.

PR106_TOTAL_RATE_DENOM = 37_545_489  # contest reference video bytes


def _toy_anchors() -> list[CalibrationAnchor]:
    """Three synthetic anchors with a clean (rate, rel_err) -> distortion curve."""
    return [
        CalibrationAnchor(
            lane_id="toy_anchor_lossless",
            rel_err_pct_per_weight=0.0,
            archive_bytes=600_000,
            contest_cuda_score=0.21,
            avg_pose_dist=3.4e-5,
            avg_seg_dist=6.7e-4,
            rate_unscaled=600_000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-04T12:00:00Z",
            job_id="toy-lossless-anchor",
            archive_sha256="0" * 64,
            notes="synthetic baseline anchor (rel_err=0)",
        ),
        CalibrationAnchor(
            lane_id="toy_anchor_int8",
            rel_err_pct_per_weight=0.40,
            archive_bytes=520_000,
            contest_cuda_score=0.29,
            avg_pose_dist=8.0e-5,
            avg_seg_dist=1.3e-3,
            rate_unscaled=520_000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-04T13:00:00Z",
            job_id="toy-int8-anchor",
            archive_sha256="1" * 64,
            notes="synthetic mid-rate anchor (mild quantization)",
        ),
        CalibrationAnchor(
            lane_id="toy_anchor_int5",
            rel_err_pct_per_weight=2.10,
            archive_bytes=440_000,
            contest_cuda_score=0.55,
            avg_pose_dist=6.0e-4,
            avg_seg_dist=4.0e-3,
            rate_unscaled=440_000 / PR106_TOTAL_RATE_DENOM,
            measured_utc="2026-05-04T14:00:00Z",
            job_id="toy-int5-anchor",
            archive_sha256="2" * 64,
            notes="synthetic aggressive-quant anchor (high rel_err)",
        ),
    ]


# ── 2. Toy distortion proxy (closed-form, CPU-only) ─────────────────────────
#
# In production this is `experiments.distortion_proxy_local.make_distortion_proxy`
# — a closed-form estimator anchored to a real archive. Here we use a trivial
# power-law to keep the example self-contained.


def _toy_distortion_proxy(
    archive_bytes: int, rel_err_pct: float, n_layers: int
) -> tuple[float, float]:
    """Return (predicted_pose_dist, predicted_seg_dist) for a toy candidate."""
    pose = 3.4e-5 + 1.5e-4 * (rel_err_pct ** 1.7)
    seg = 6.7e-4 + 1.0e-3 * (rel_err_pct ** 1.3)
    return pose, seg


# ── 3. Synthetic candidates to rank ─────────────────────────────────────────


def _toy_candidates(stub_archive: Path) -> list[dict]:
    """Five synthetic codec-parameter candidates spanning the (bytes, rel_err) frontier.

    `stub_archive` is the placeholder file path the sanity gate will inspect.
    In production each candidate has its own real archive on disk.
    """
    return [
        # near-baseline; should rank well
        {
            "candidate_id": "cand_int8_baseline",
            "archive_bytes": 520_000,
            "rel_err_pct": 0.40,
            "n_layers": 8,
            "lane_class": "apogee_intN",
            "archive_path": stub_archive,
        },
        # tighter rate, slightly higher rel_err — modest predicted score
        {
            "candidate_id": "cand_int7_balanced",
            "archive_bytes": 480_000,
            "rel_err_pct": 0.95,
            "n_layers": 8,
            "lane_class": "apogee_intN",
            "archive_path": stub_archive,
        },
        # aggressive quantization inside calibration range — predictor still works
        {
            "candidate_id": "cand_int5_aggressive",
            "archive_bytes": 440_000,
            "rel_err_pct": 2.05,
            "n_layers": 8,
            "lane_class": "apogee_intN",
            "archive_path": stub_archive,
        },
        # extrapolation regime — predictor SHOULD widen / refuse
        {
            "candidate_id": "cand_int4_extrapolation",
            "archive_bytes": 410_000,
            "rel_err_pct": 7.10,  # past the calibration max (2.10)
            "n_layers": 8,
            "lane_class": "apogee_intN",
            "archive_path": stub_archive,
        },
        # tiny bytes but proxy says distortion is huge — should fail Lagrangian
        {
            "candidate_id": "cand_int3_distortion_blowup",
            "archive_bytes": 380_000,
            "rel_err_pct": 12.0,
            "n_layers": 8,
            "lane_class": "apogee_intN",
            "archive_path": stub_archive,
        },
    ]


# ── 4. Mocked parallel-dispatch actuator ────────────────────────────────────
#
# The May 4 race-window failure mode (per docs/paper/07_discussion.md §7.8) was
# building the ranker without an *actuator* that fans out N concurrent paid-GPU
# dispatches. The shape below is what `tools/parallel_dispatch_top_k.py` does
# in production, with the dispatch replaced by a 0.1-second sleep so the
# example runs without spending a cent.


def _mock_dispatch_one(candidate_id: str) -> dict:
    """Pretend to dispatch a candidate; sleep briefly to simulate wall-clock."""
    time.sleep(0.1)
    # Simulated harvested score — close to but not equal to the predicted band.
    return {
        "candidate_id": candidate_id,
        "harvested_score": 0.21 + hash(candidate_id) % 17 / 200.0,
        "tag": "[contest-CUDA]",  # in production: enforce this; drop rows without it
    }


def _mock_parallel_dispatch(candidate_ids: Iterable[str], max_workers: int = 4) -> list[dict]:
    """Fan out N concurrent (mocked) dispatches and collect harvested rows."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_mock_dispatch_one, cid): cid for cid in candidate_ids}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


# ── 5. Main quickstart ──────────────────────────────────────────────────────


def main() -> None:
    # Step 1: build the search engine with toy anchors + toy proxy.
    anchors = _toy_anchors()
    constraints = LagrangianConstraints(
        rate_unscaled_max=520_000 / PR106_TOTAL_RATE_DENOM,
        pose_dist_max=2e-4,
        seg_dist_max=2e-3,
    )
    # Inject a permissive sanity gate so the toy walk-through makes it past
    # gate 4 — production wires `tools/predispatch_sanity.py predispatch_sanity`
    # which checks 5 archive-on-disk invariants before any paid dispatch.
    from types import SimpleNamespace

    def _toy_sanity_gate(**_kwargs):
        return SimpleNamespace(passed=True, refusal_reasons=[], gates=[])

    search = MetaLagrangianSearch(
        calibration_anchors=anchors,
        distortion_proxy=_toy_distortion_proxy,
        constraints=constraints,
        sanity_gate=_toy_sanity_gate,
    )

    # Step 2: rank all candidates, then pull the top-3 dispatch-eligible ones.
    # The sanity gate inspects an `archive_path`; in this toy run we point
    # every candidate at a stub file (this script itself) — the override above
    # makes the gate permissive so the demo can fan out without producing
    # real archives. Production: each candidate has its own real archive.zip.
    stub_archive = Path(__file__).resolve()
    candidates = _toy_candidates(stub_archive)
    ranked = search.evaluate_all(candidates)
    top_3 = MetaLagrangianSearch.top_k(ranked, k=3)

    print("=== STEP 2 — Lagrangian ranking of 5 toy candidates ===")
    for ev in ranked:
        status = "DISPATCH" if ev.eligible_for_dispatch else "REFUSED"
        reason = ev.band_refusal_reason or ",".join(ev.sanity_failures) or "-"
        print(
            f"  {ev.candidate_id:<35s} L={ev.lagrangian:8.4f} "
            f"band=[{ev.band_low:6.4f},{ev.band_high:6.4f}] "
            f"{status:<8s} {reason[:60]}"
        )

    print("\n=== STEP 3 — Predictor refusal modes ===")
    # Show INSUFFICIENT_ANCHORS by passing only one anchor.
    band_few = predict_score_band(
        archive_bytes=440_000,
        rel_err_pct_per_weight=2.10,
        n_quantized_layers=8,
        calibration_anchors=anchors[:1],
        distortion_proxy=_toy_distortion_proxy,
    )
    print(f"  insufficient_anchors -> {band_few.as_str()}")

    # Show EXTRAPOLATION refusal at rel_err well above the anchor max.
    band_extrap = predict_score_band(
        archive_bytes=410_000,
        rel_err_pct_per_weight=7.10,
        n_quantized_layers=8,
        calibration_anchors=anchors,
        distortion_proxy=None,  # also triggers HIGH_REL_ERR_WITHOUT_PROXY
    )
    print(f"  extrapolation+no_proxy -> {band_extrap.as_str()}")

    # In-calibration request — should NOT refuse.
    band_ok = predict_score_band(
        archive_bytes=520_000,
        rel_err_pct_per_weight=0.40,
        n_quantized_layers=8,
        calibration_anchors=anchors,
        distortion_proxy=_toy_distortion_proxy,
    )
    print(f"  in-calibration       -> {band_ok.as_str()}")

    # Step 4: parallel-dispatch + harvest-and-reseed (mocked).
    print("\n=== STEP 4 — Parallel-dispatch actuator (MOCKED — no real GPU) ===")
    if not top_3:
        print("  No dispatch-eligible candidates; nothing to fan out.")
        return

    print(f"  Fanning out {len(top_3)} concurrent dispatches via ThreadPoolExecutor...")
    t0 = time.perf_counter()
    harvested = _mock_parallel_dispatch([ev.candidate_id for ev in top_3])
    wall_clock = time.perf_counter() - t0
    print(f"  Wall-clock: {wall_clock:.2f}s for {len(harvested)} concurrent dispatches.")
    for row in harvested:
        print(f"    {row['candidate_id']:<35s} score={row['harvested_score']:.4f} {row['tag']}")

    # Step 5: harvest-and-reseed sketch — drop rows without [contest-CUDA] tag,
    # build a new anchor for next loop iteration.
    print("\n=== STEP 5 — Harvest + reseed sketch ===")
    new_anchors = []
    for row in harvested:
        if row["tag"] != "[contest-CUDA]":
            print(f"  DROPPED {row['candidate_id']} (untagged or wrong device — auth-eval-everywhere rule)")
            continue
        # In production: cross-verify against per-dispatch contest_auth_eval.json.
        # Here we just sketch the append.
        ev = next(e for e in top_3 if e.candidate_id == row["candidate_id"])
        new_anchors.append(
            CalibrationAnchor(
                lane_id=row["candidate_id"],
                rel_err_pct_per_weight=ev.rel_err_pct,
                archive_bytes=ev.archive_bytes,
                contest_cuda_score=row["harvested_score"],
                avg_pose_dist=ev.proxy_pose,
                avg_seg_dist=ev.proxy_seg,
                rate_unscaled=ev.proxy_rate_unscaled,
                measured_utc="2026-05-05T17:00:00Z",
                job_id=f"toy-harvest-{row['candidate_id']}",
                archive_sha256="9" * 64,
                notes="synthetic harvested anchor; production: cross-verify contest_auth_eval.json",
            )
        )
    print(f"  Harvested {len(new_anchors)} new anchors; ready to seed loop iteration N+1.")
    print("\nDone. In production this loop runs end-to-end via tools/feedback_loop_sweep.py.")


if __name__ == "__main__":
    main()
