# Changelog

All notable changes to `tac` (Task-Aware Codec) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.5] — 2026-05-05 (Initial public release)

### Added
- Initial public OSS release on PyPI as part of the comma video compression
  challenge submission cycle (PR #107 `apogee`, 0.2293 contest-CUDA T4).
- 8 post-filter architectures (Standard, Dilated, PixelShuffle, PSD,
  Depthwise, Luma, FiLM, PairAware) with 12 variant aliases.
- `tac.training.Trainer` with QAT, EMA, SWA, best-checkpoint selection,
  lazy data loading, and resume support.
- `tac.optimizer.MetaLagrangianSearch` — Boyd-style multi-constraint search
  combining a closed-form distortion proxy, a score-band predictor, and a
  5-gate predispatch sanity ladder.
- `tac.predictor.score_band` — score-band predictor with explicit refusal
  modes (`insufficient_anchors`, `extrapolation`,
  `lossy_better_than_lossless_incoherent`, ...). Built after the apogee_int4
  8× miss; refusal is the feature, not the bug.
- `tac.preflight` — strict-mode preflight checks (~50+ structural invariants)
  that catch dispatch-time hazards before paid GPU spend.
- `tac.fp4_quantize` — extreme 4-bit quantization with codebook.
- `tac.mask_codec` — mask extraction, AV1/VVC encoding, entropy coding.
- `tac.renderer` — neural mask-to-RGB renderer for the GPU lane.
- `tac.tto` — test-time optimization at inflation.
- `tac.scorer` — scoring formula, sensitivity analysis.
- `tac.evaluate` — proxy evaluation, top-K checkpoint averaging.
- GitHub Actions CI matrix on Python 3.11 / 3.12 with contest-score formula
  constants verification.
- `examples/quickstart.py` — meta-Lagrangian search + closed-loop feedback
  walk-through on a TOY synthetic problem (no GPU, no comma archives needed).

### Context
This release coincides with the post-mortem of the comma video compression
challenge (deadline 2026-05-04 12:00 UTC, our final submission PR #107
`apogee` 0.2293, ranked ~11th). The closed-loop search primitives shipped in
this version were built during and immediately after the May 4 race window
that decided the contest. The full lessons-learned writeup — including the
"planner-without-actuator" failure mode that motivated the parallel-dispatch
actuator — appears in `docs/paper/07_discussion.md` §7.8 in the parent
`comma-lab` research repository.

[1.0.5]: https://github.com/adpena/tac/releases/tag/v1.0.5
