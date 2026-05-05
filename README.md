# tac -- Task-Aware Codec

Neural video compression optimized for downstream perception models.

tac trains tiny CNN post-filters that correct decoded video frames by backpropagating through frozen perception networks. The filter learns corrections that minimize the scorer's distortion metric, not generic pixel quality.

## Installation

```bash
pip install tac
```

Optional extras:

```bash
pip install tac[mlx]        # Apple Silicon acceleration
pip install tac[viz]         # Plotly visualization
pip install tac[notebooks]   # Marimo notebook support
```

## Quick start

```python
from tac import Trainer, TrainConfig, build_postfilter

# Build a 3-layer residual CNN post-filter
model = build_postfilter("standard", hidden=64)

# Configure training with QAT, EMA, and best-checkpoint selection
config = TrainConfig(hidden=64, epochs=1000, alpha=20, tag="my_run")

# Train against frozen PoseNet + SegNet scorers
trainer = Trainer(model, config, device="mps")
trainer.fit(comp_pairs, gt_pairs, posenet, segnet, sal_weights)
```

For the closed-loop search side of the library — meta-Lagrangian ranker,
score-band predictor with refusal modes, parallel-dispatch actuator, and
harvest-and-reseed feedback — see [`examples/quickstart.py`](examples/quickstart.py).
It runs without a GPU and without comma-challenge data:

```bash
python examples/quickstart.py
```

## What this is for

**Meta-Lagrangian search engine** (`tac.optimizer.MetaLagrangianSearch`):
ranks codec candidates with a Boyd-style multi-constraint Lagrangian,
combining a closed-form distortion proxy, a score-band predictor, and a
5-gate predispatch sanity ladder. Refused candidates sort to the bottom of
the dispatch queue regardless of nominal score; the engine is deterministic,
arch-agnostic, and uses the exact contest score formula.

**Predictor with refusal modes** (`tac.predictor.score_band`): predicts a
contest-CUDA score band from rel_err and archive bytes, but *refuses* when
calibration support is insufficient (`insufficient_anchors`,
`extrapolation`, `lossy_better_than_lossless_incoherent`, ...). Built after
the apogee_int4 8x miss (predicted [0.155, 0.180]; landed 1.4287
[contest-CUDA]) — refusal is the feature, not the bug.

**Parallel-dispatch actuator** (production wires
`tools/parallel_dispatch_top_k.py` from the parent comma-lab repo):
`concurrent.futures.ThreadPoolExecutor` over the existing dispatch wrappers
with per-dispatch and total-cost gating, harvested-JSONL output, and strict
refusal of candidates not marked `ready_for_exact_eval_dispatch=true`.

**Closed-loop feedback** (production wires
`tools/feedback_loop_sweep.py`): rank → fan-out N concurrent paid-GPU
dispatches → harvest [contest-CUDA] rows → cross-verify against
`contest_auth_eval.json` → append empirical anchors → re-rank → repeat,
gated by `--max-cycles`, `--max-total-cost`, `--max-cost-per-cycle`, and
`--convergence-eps`.

## What it does

Video compression codecs (H.264, AV1) optimize for human perception -- PSNR, SSIM, perceptual quality. But many downstream consumers are neural networks, not humans. A self-driving car's perception stack does not care about perceptual quality; it cares about whether PoseNet and SegNet produce correct outputs from the decoded frames.

tac bridges this gap:

1. **Compress** video with a standard codec (H.265, AV1)
2. **Post-filter** decoded frames through a tiny learned CNN
3. **Score** using the actual downstream perception models
4. **Backpropagate** through the frozen scorers to train the filter

The post-filter learns artifact corrections that specifically help the downstream models, even if those corrections look invisible (or worse) to human eyes.

## Architecture

tac provides two processing lanes:

- **CPU lane**: Standard codec + learned post-filter. The post-filter is a 3-layer residual CNN (~390KB int8) that runs in real-time on CPU.
- **GPU lane**: Mask extraction + neural rendering. SegNet masks are compressed at extreme ratios, then a neural renderer reconstructs RGB frames from masks alone.

### Key modules

| Module | Purpose |
|---|---|
| `tac.architectures` | 8 post-filter architectures (Standard, Dilated, PixelShuffle, PSD, ...) |
| `tac.training` | Trainer with QAT, EMA, SWA, best-checkpoint selection |
| `tac.losses` | Scorer-aware losses (standard, feature matching, STE) |
| `tac.quantization` | Int8 quantization (per-channel, FakeQuant STE, LSQ) |
| `tac.fp4_quantize` | Extreme 4-bit quantization with codebook |
| `tac.mask_codec` | Mask extraction, AV1/VVC encoding, entropy coding |
| `tac.renderer` | Neural mask-to-RGB renderer (GPU lane) |
| `tac.tto` | Test-time optimization at inflation |
| `tac.scorer` | Scoring formula, sensitivity analysis |
| `tac.evaluate` | Proxy evaluation, checkpoint averaging |
| `tac.profiles` | Named training profiles (proven_baseline, smoke, ...) |

## CLI

```bash
# Train a post-filter
tac lossy train --profile proven_baseline --precomputed data/precomputed

# Evaluate a checkpoint
tac lossy eval --checkpoint best_int8.pt --archive test.zip

# Lossless compression tools
tac lossless compress input.bin -o output.tac
tac lossless decompress output.tac -o recovered.bin
```

## Links

- [Repository](https://github.com/adpena/pact)
- [comma.ai video compression challenge](https://github.com/commaai/comma-video-compression-challenge)

## License

MIT

## Comma Video Compression Challenge

This library powers our submission to the
[comma video compression challenge](https://github.com/commaai/comma_video_compression_challenge),
PR [#107 (apogee, 0.2293 contest-CUDA T4)](https://github.com/commaai/comma_video_compression_challenge/pull/107).

Key modules used in the submission:

- `tac.optimizer.MetaLagrangianSearch` — Boyd-style multi-constraint search
  integrating predictor + distortion proxy + 5-gate sanity ladder. Refused
  candidates rank to the bottom of the dispatch queue regardless of nominal
  score. Engine is deterministic, uses the exact contest score formula
  (`100*seg + sqrt(10*pose) + 25*archive_bytes/37545489`), and is
  arch-agnostic (accepts arbitrary calibration anchors).

- `tac.predictor.score_band` — score band predictor with explicit refusal
  modes (`insufficient_anchors`, `extrapolation`,
  `lossy_better_than_lossless`, etc.). Refuses outside its calibration
  range rather than extrapolating.

- `tac.preflight` — strict-mode preflight checks (~50+ structural
  invariants) that catch dispatch-time hazards before paid GPU spend.

The full research workspace (training scripts, dispatch tooling, ledgers)
lives in a separate private repository pending sanitization for OSS
release.
