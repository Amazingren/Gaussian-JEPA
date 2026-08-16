# Results

This directory records the final paper protocols and results for the released
300-epoch, 1K-Gaussian `E(All)` checkpoint.

| Document | Evaluation |
|---|---|
| [classification.md](classification.md) | ModelNet10/40 Full, Linear, and MLP-3 transfer |
| [partseg.md](partseg.md) | ShapeNet-Part segmentation |
| [resampling_consistency.md](resampling_consistency.md) | Frozen consistency under independent Gaussian resampling |
| [partial_observation.md](partial_observation.md) | Frozen retrieval from spatially partial observations |
| [completion.md](completion.md) | Frozen-feature Gaussian shape completion |

Reported Gaussian comparisons use the same sampled inputs and protocol within
each experiment. Raw logs, checkpoints, and generated artifacts are intentionally
excluded from the release repository.
