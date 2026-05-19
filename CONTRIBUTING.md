# Contributing to tac

Thank you for considering a contribution. This document follows the
[comma.ai openpilot](https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md)
contribution conventions.

By submitting a contribution, you agree that:

1. Your contribution is licensed under the **MIT License** (see `LICENSE`).
2. You have the right to submit the contribution under that license.
3. You consent to having `Copyright (c) 2026 Alejandro (Alex) Peña` retained as
   the project copyright; your individual authorship is recorded in git history.

## Code of conduct

Be excellent to each other. Bad-faith contributions (license violations,
plagiarism, harassment, malicious code) are not welcome.

## Naming rule

`tac` means **Task-Aware Compression**: the reusable compression library and
algorithmic engine. Use `codec` only for concrete encoders, decoders, entropy
coders, archive grammars, or wire formats inside that broader stack.

## Development workflow

```bash
# Clone
git clone https://github.com/adpena/tac.git
cd tac

# Install (editable + test deps)
python3 -m venv .venv
.venv/bin/pip install -e . pytest pytest-timeout hypothesis ruff

# Run the deterministic CPU-only test suite
.venv/bin/python -m pytest tac/tests/test_meta_lagrangian.py \
                            tac/tests/test_predictor_score_band.py \
                            tac/tests/test_distortion_proxy_local.py \
                            -v --timeout=60

# Lint
.venv/bin/ruff check tac/

# Verify the contest score formula reproduces (PR106 + apogee_int4)
.venv/bin/python -c "
from tac.optimizer import contest_score
assert abs(contest_score(3.4e-5, 0.00067819, 186239) - 0.20945673) < 5e-3
assert abs(contest_score(0.02370903, 0.00868503, 109996) - 1.42866394) < 1e-3
print('Formula reproductions verified.')
"
```

## Pull request expectations

- One logical change per PR.
- Tests for new behavior.
- No regressions in the existing test suite.
- Commit messages follow `<what changed>: <why>` style.
- Score-affecting trainer / codec / archive changes need either an empirical
  anchor (CUDA auth eval JSON), a derivation, or a
  `[provenance-only; no score claim]` tag.

## What we do NOT accept

- Hard-coded local absolute paths (e.g. `/Users/<name>/...`, `/home/<name>/...`,
  `/tmp/...` as durable evidence). Use placeholders or repo-relative paths.
- Score claims based on MPS, local CPU scorers, or non-Linux-x86_64 CPU eval
  without explicit `[advisory only]` tagging.
- Dependencies that are GPL / AGPL in the default install path. The
  LGPL-2.1-or-later `pyppmd` dependency in this project is opt-in via the
  parent research workspace's `[pr86_replay]` extra and is documented there.

## Related repositories

- **Research workspace**: [adpena/comma-lab](https://github.com/adpena/comma-lab)
  contains the full experimental ledger, public-PR intake, dispatch tooling,
  and methodology writeups. `tac` is the curated production extract.
- **Upstream challenge**: [commaai/comma_video_compression_challenge](https://github.com/commaai/comma_video_compression_challenge)
  defines the scoring formula, evaluator, and submission contract this library
  targets.

## License

By contributing, you license your work under the MIT License. See `LICENSE`.

`Copyright (c) 2026 Alejandro (Alex) Peña`
