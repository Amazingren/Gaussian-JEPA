# Checkpoint format

Pretrained weights are not included in the current public release. This page
documents the reference checkpoint layout so that locally supplied weights can
be used consistently. No baseline or third-party weights are distributed with
the repository.

## Pretrained model

The reference encoder was pretrained for 300 epochs on ShapeNet55-GS with 1,024
Gaussians per object.

| File | Pretraining class | Epoch | Size | SHA-256 |
|---|---|---:|---:|---|
| `pretrain/gaussian_jepa_shapenet55gs_1k_ep300.pth` | `GaussianJEPA` | 300 | 380,976,155 bytes | `42e753944cc8e0e763df4bf1c59f465a1f368b5a53c9f87493ec0c07f3eee5bc` |

The checkpoint stores 377 tensors in `checkpoint["base_model"]`. Online
encoder parameters use the prefix `JEPA_encoder.`. The concise public registry
names `GaussianJEPA` and `PointTransformer_GaussianJEPA` are aliases for the
legacy class names stored by the training code, so the reference checkpoint is
loaded without key conversion.

Verify a local copy before use:

```bash
sha256sum pretrain/gaussian_jepa_shapenet55gs_1k_ep300.pth
```

The README commands use `checkpoints/gaussian_jepa_ep300.pth` as a concise
local alias. Copy the pretrained file to that path before running them:

```bash
mkdir -p checkpoints
cp pretrain/gaussian_jepa_shapenet55gs_1k_ep300.pth \
  checkpoints/gaussian_jepa_ep300.pth
```

Classification configurations automatically extract `JEPA_encoder.*` when
this checkpoint is supplied.

## Downstream models

Downstream checkpoints may use the same task-specific wrappers documented for
ModelNet classification, ShapeNet-Part segmentation, and ShapeNet55-GS
completion. Their filenames should encode the task, transfer protocol, Gaussian
budget, and training seed.
