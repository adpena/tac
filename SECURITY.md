# Security policy

## Reporting a vulnerability

If you discover a security issue in the `tac` library, please report it
privately rather than opening a public issue.

Contact: `adpena@users.noreply.github.com` (GitHub-routed alias).

When reporting, please include:

- a brief description of the issue and its impact;
- a minimal reproduction (if possible);
- the affected file, function, or commit SHA;
- any suggested remediation.

We aim to acknowledge security reports within 7 days and to issue a patch or
mitigation guidance within 30 days for confirmed vulnerabilities. Public
disclosure should be coordinated to give downstream consumers time to upgrade.

## Scope

In scope:

- the `tac` Python package (top-level `tac/`);
- example code under `examples/`;
- CI workflows under `.github/workflows/`.

Out of scope:

- third-party dependencies (report to their respective maintainers);
- the [comma video compression challenge](https://github.com/commaai/comma_video_compression_challenge)
  upstream snapshot (report there).

## Supply-chain considerations

This package pins hard runtime dependencies to major-version ranges. The
default `pip install tac` install path is permissive-only (MIT / Apache-2.0
/ BSD-3-Clause / tri-licensed MIT/Apache-2.0/BSL-1.0 for `constriction`).
The `pyppmd` LGPL-2.1-or-later dependency is documented in
[the parent research workspace](https://github.com/adpena/comma-lab).
