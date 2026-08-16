# Contributing

Gaussian-JEPA is a research codebase. Focused bug reports and changes that
improve reproducibility, documentation, or compatibility with the released
checkpoint are welcome.

## Before opening an issue

Please search existing issues and run the release smoke test when the problem
concerns model loading or CUDA operators:

```bash
python tools/smoke_test_release.py \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth
```

A useful report includes:

- the exact command and configuration;
- Python, PyTorch, CUDA toolkit, and GPU versions;
- the checkpoint SHA-256 digest;
- the complete error traceback or relevant log excerpt;
- a minimal description of the input schema, without redistributing gated
  dataset assets.

## Changes

Keep pull requests narrow and preserve compatibility with the canonical
`gaussian_jepa_ep300.pth` checkpoint. New experiments should not modify final
paper protocols silently; add a separate configuration and document the
changed assumption. Run Python syntax checks, parse edited YAML files, and
execute the smallest relevant GPU smoke test before submitting a change.

By contributing, you agree that your changes are distributed under this
repository's license and retain applicable third-party notices.
