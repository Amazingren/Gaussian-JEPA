# Gaussian-JEPA model card

## Model summary

Gaussian-JEPA is a self-supervised encoder for object-level 3D Gaussian
Splatting (3DGS) assets. It learns by predicting latent features of spatially
held-out Gaussian token blocks from their visible context. The released model
uses a 12-layer, 384-dimensional transformer and was pretrained for 300 epochs
on ShapeNet55-GS.

| Item | Released setting |
|---|---|
| Checkpoint | `gaussian_jepa_ep300.pth` |
| Input budget | 1,024 Gaussians |
| Per-Gaussian features | xyz, opacity, scale, quaternion, and degree-0 SH (14 values) |
| Tokenization | 64 xyz-centered groups, 32 neighbors per group |
| Encoder output | 64 tokens of 384 dimensions |
| Global representation | L2-normalized concatenated mean/max token pooling (768-D) |
| Pretraining data | ShapeNet55-GS training split |
| License | CC BY-SA 4.0; see `THIRD_PARTY_NOTICES.md` |

The file digest and checkpoint structure are documented in
[`CHECKPOINTS.md`](CHECKPOINTS.md).

## Intended use

The checkpoint is intended for research on object-level Gaussian
representations, including:

- frozen-feature analysis and retrieval;
- transfer to object classification and part segmentation;
- representation-based Gaussian shape completion;
- controlled studies of resampling and partial observations.

The encoder produces representations, not rendered images or class labels by
itself. Task-specific prediction heads must be trained separately.

## Input contract

Input PLY files must contain the standard 3DGS fields used by the ShapeSplats
release: `x/y/z`, `opacity`, `scale_0..2`, `rot_0..3`, and `f_dc_0..2`.
Coordinates are centered and normalized to the unit sphere; physical scales
are divided by the same radius. Opacity is sigmoid-activated, rotations are
unit-normalized, and degree-0 spherical-harmonic coefficients provide the
appearance channels.

Use [`tools/extract_features.py`](tools/extract_features.py) for a reproducible
single-object inference example.

## Evaluation

The released checkpoint is evaluated on ModelNet10/40-GS classification,
ShapeNet-Part segmentation, Gaussian resampling consistency, partial-to-complete
retrieval, and Gaussian shape completion. Exact protocols and results are in
[`docs/results`](docs/results/README.md).

## Scope and limitations

- The model was developed for isolated object assets, not large scenes or
  unconstrained temporal captures.
- Group construction uses centroid xyz distances. Anisotropic Gaussian support
  is encoded as an attribute but is not used to define neighborhoods.
- The reference pipeline uses a fixed 1K-Gaussian input budget. Other budgets
  may require protocol-specific validation.
- Results depend on the 3DGS fitting process, attribute schema, and downstream
  data distribution. The model has not been validated for safety-critical or
  production decision systems.

These boundaries describe the validated release rather than prohibiting
research extensions to scenes, alternative groupings, or larger input budgets.

## Reproducibility

Verify the checkpoint SHA-256 digest, run the dataset-free GPU smoke test, and
record the sampling seed used for feature extraction:

```bash
python tools/smoke_test_release.py \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth

python tools/extract_features.py \
  --ply /path/to/point_cloud.ply \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth \
  --output outputs/object_features.npz \
  --seed 0
```

## Citation

The archival paper citation will be added when the public manuscript is
available. Until then, please cite the repository and retain the license and
upstream attribution notices.
