# Released checkpoints

The [Gaussian-JEPA checkpoint folder](https://drive.google.com/drive/folders/1OEXX2ZWsnoL0h4Bzf2sURBGJeneI-C3N?usp=sharing)
contains the canonical pretrained model and the principal downstream models.
No baseline or third-party weights are included.

## Pretrained model

The released encoder was pretrained for 300 epochs on ShapeNet55-GS with 1,024
Gaussians per object.

| File | Pretraining class | Epoch | Size | SHA-256 |
|---|---|---:|---:|---|
| `pretrain/gaussian_jepa_shapenet55gs_1k_ep300.pth` | `GaussianJEPA` | 300 | 380,976,155 bytes | `42e753944cc8e0e763df4bf1c59f465a1f368b5a53c9f87493ec0c07f3eee5bc` |

The checkpoint stores 377 tensors in `checkpoint["base_model"]`. Online
encoder parameters use the prefix `JEPA_encoder.`. The concise public registry
names `GaussianJEPA` and `PointTransformer_GaussianJEPA` are aliases for the
legacy class names stored by the training code, so the released checkpoint is
loaded without key conversion.

Verify the downloaded file before use:

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

The same folder also provides six ModelNet classifiers, the ShapeNet-Part
model associated with the reported 84.5 class mIoU / 86.1 instance mIoU, and
three independently trained ShapeNet55-GS completion decoders. Their filenames
encode the task, transfer protocol, Gaussian budget, and reported result. Each
task retains its native training wrapper; use the corresponding configuration
and evaluation entry point documented in the repository.
