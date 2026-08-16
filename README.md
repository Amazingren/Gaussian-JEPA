<div align="center">

# Gaussian-JEPA

### Joint-Embedding Predictive Learning for 3D Gaussian Splats

**<a href="https://amazingren.github.io/">Bin Ren</a><sup>1</sup>,
<a href="https://qimaqi.github.io/">Qi Ma</a><sup>2</sup>,
<a href="https://unique1i.github.io/">Yue Li</a><sup>3</sup>,
<a href="https://scholar.google.com/citations?user=rL9CaNwAAAAJ&hl=en&oi=ao">Zongyan Han</a><sup>1</sup>,
<a href="https://liyidi.github.io/">Yidi Li</a><sup>4</sup>,
<a href="https://scholar.google.com/citations?user=y3Bpp1IAAAAJ&hl=en">Yuqian Fu</a><sup>5</sup>**<br>
**<a href="https://scholar.google.com/citations?user=_KlvMVoAAAAJ&hl=en">Rao Muhammad Anwer</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=yqsvxQgAAAAJ&hl=en">Theo Gevers</a><sup>3</sup>,
<a href="https://scholar.google.com/citations?user=zvaeYnUAAAAJ&hl=en">Fahad Shahbaz Khan</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en">Salman Khan</a><sup>1</sup>**

<sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) &nbsp;&nbsp;
<sup>2</sup>ETH Zürich &nbsp;&nbsp;
<sup>3</sup>University of Amsterdam<br>
<sup>4</sup>Taiyuan University of Technology &nbsp;&nbsp;
<sup>5</sup>King Abdullah University of Science and Technology (KAUST)

[![Project Page](https://img.shields.io/badge/Project-Page-176B63?logo=googlechrome&logoColor=white)](https://amazingren.github.io/Gaussian-JEPA/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/Amazingren/Gaussian-JEPA)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Google_Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1OEXX2ZWsnoL0h4Bzf2sURBGJeneI-C3N?usp=sharing)
[![Documentation](https://img.shields.io/badge/Docs-Getting_Started-176B63?logo=readthedocs&logoColor=white)](docs/getting_started.md)
<br>
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&logoColor=white)](environment.yml)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](environment.yml)
[![License](https://img.shields.io/badge/License-CC_BY--SA_4.0-8A2BE2)](LICENSE)
<br>
[![Pretraining](https://img.shields.io/badge/Task-Pretraining-2A9D8F)](#pretraining)
[![Classification](https://img.shields.io/badge/Task-Classification-457B9D)](docs/results/classification.md)
[![Part Segmentation](https://img.shields.io/badge/Task-Part_Segmentation-E9A23B)](docs/results/partseg.md)
[![Shape Completion](https://img.shields.io/badge/Task-Shape_Completion-C05A7A)](docs/results/completion.md)

<!-- Add the public arXiv badge to the first row once its identifier is available. -->

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/Amazingren/Gaussian-JEPA/main/assets/gaussian_jepa_teaser.png" width="900"
       alt="Gaussian-JEPA motivation and learning paradigms">
</p>

## Overview

Gaussian-JEPA is a self-supervised framework for learning object-level
representations directly from 3D Gaussian Splatting (3DGS) assets. It replaces
masked-attribute reconstruction with latent prediction: an online encoder
represents visible Gaussian tokens, while an exponential-moving-average target
encoder provides stop-gradient supervision for held-out spatial blocks.

The released model uses all 14 Gaussian attributes, four non-overlapping target
blocks with heterogeneous spatial support, complementary latent projections,
and feature-space grounding. The repository provides the pretraining
implementation, a 300-epoch checkpoint, downstream transfer protocols, frozen
representation diagnostics, shape completion, and part segmentation.

## Method

<p align="center">
  <img src="https://raw.githubusercontent.com/Amazingren/Gaussian-JEPA/main/assets/gaussian_jepa_framework.png" width="900"
       alt="Gaussian-JEPA framework">
</p>

A 1K-Gaussian observation is grouped into 64 local tokens. The target sampler
selects four non-overlapping blocks of sizes `[11, 9, 7, 5]`; their complement
forms a shared 32-token context. The context encoder runs once, after which a
shared predictor estimates two latent projections for each target block from
features supplied by the EMA encoder. Training remains entirely in
representation space and does not decode coordinates, opacity, covariance, or
spherical-harmonic coefficients.

## Results

Gaussian classification results below use 1K Gaussians for both pretraining
and transfer. Detailed settings and complete metrics are available in
[`docs/results`](docs/results/README.md).

| Evaluation | Gaussian-MAE | Gaussian-JEPA |
|:--|--:|--:|
| ModelNet10 linear probing (%) | 93.50 | **93.72** |
| ModelNet40 linear probing (%) | 88.97 | **90.47** |
| ShapeNet-Part class mIoU (%) | 84.2 | **84.5** |
| Resampling drift ↓ | 0.1202 | **0.0850** |
| Partial-observation R@1 AUC ↑ | 41.82 | **52.74** |
| Shape-completion Chamfer distance ↓ | 0.0732 | **0.0678** |

## Installation

The reference environment uses Python 3.9, PyTorch 2.0.1, and CUDA 11.8.
`pointnet2_ops` and `knn_cuda` are compiled during installation, so the CUDA
toolkit exposed by `nvcc` must be compatible with the installed PyTorch build.

```bash
git clone https://github.com/Amazingren/Gaussian-JEPA.git
cd Gaussian-JEPA
conda env create -f environment.yml
conda activate gaussian_jepa
```

See [`docs/getting_started.md`](docs/getting_started.md) for dataset layouts,
configuration details, and the complete evaluation workflow.

## Data

Pretraining uses
[ShapeSplatsV1](https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1), while
classification uses
[ModelNet Splats](https://huggingface.co/datasets/ShapeSplats/ModelNet_Splats).
Request access and accept the corresponding dataset terms before downloading.
Dataset locations can be configured without modifying tracked source files:

```bash
export SHAPENET55GS_PLY_ROOT=/path/to/shapesplat_ply
export MODELNETGS_PLY_ROOT=/path/to/modelsplat_ply
```

The ShapeNet55 and ModelNet split files used by the loaders are included under
[`datasets/`](datasets/).

## Released checkpoints

The [Gaussian-JEPA checkpoint folder](https://drive.google.com/drive/folders/1OEXX2ZWsnoL0h4Bzf2sURBGJeneI-C3N?usp=sharing)
contains the canonical pretrained model and the principal classification,
part-segmentation, and shape-completion checkpoints. For the commands below,
download `pretrain/gaussian_jepa_shapenet55gs_1k_ep300.pth` and save it locally
as `checkpoints/gaussian_jepa_ep300.pth`.

```bash
mkdir -p checkpoints
sha256sum checkpoints/gaussian_jepa_ep300.pth
python tools/smoke_test_release.py \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth
```

The expected digest and checkpoint structure are documented in
[`CHECKPOINTS.md`](CHECKPOINTS.md). The smoke test verifies CUDA extensions,
strict checkpoint loading, finite pretraining losses, and one backward pass
without requiring a dataset.

## Pretraining

```bash
python main.py \
  --config cfgs/pretrain/gaussian_jepa.yaml \
  --exp_name gaussian_jepa \
  --launcher none \
  --seed 0
```

The canonical configuration trains for 300 epochs with 1,024 Gaussians, 64
groups of 32 neighbors, and a total batch size of 256. For distributed
training, launch `main.py` with `torchrun` and set `--launcher pytorch`.

## Downstream evaluation

### ModelNet classification

Six configurations cover full fine-tuning, linear probing, and MLP-3 probing
on ModelNet10 and ModelNet40. For example:

```bash
python main.py \
  --config cfgs/finetune/modelnet10_linear.yaml \
  --finetune_model \
  --ckpts checkpoints/gaussian_jepa_ep300.pth \
  --exp_name modelnet10_linear \
  --seed 0
```

Select another protocol by changing the configuration filename. The transfer
loader extracts the released `JEPA_encoder.*` weights automatically.

### Frozen Gaussian diagnostics

The resampling and partial-observation evaluators give Gaussian-JEPA and a
compatible Gaussian-MAE E(All) checkpoint identical primitives, groupings, and
observation masks.

```bash
python tools/eval_frozen_embeddings.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt /path/to/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/resampling

python tools/eval_partial_retrieval.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt /path/to/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/partial_observation
```

`--num-objects 0` evaluates all 2,467 ModelNet40-GS test objects. A positive
value runs a category-balanced subset for a faster diagnostic.

### Shape completion and part segmentation

- [`completion_gs/`](completion_gs/README.md) contains frozen-encoder,
  partial-to-complete Gaussian prediction and render-space evaluation.
- [`segmentation_gs/`](segmentation_gs/README.md) contains ShapeNet-Part
  fine-tuning and evaluation.
- [`viz/`](viz/README.md) contains the rendering and qualitative-analysis
  utilities used by the release.

## Feature extraction

The input PLY must follow the standard 3DGS property convention used by
ShapeSplats.

```bash
python tools/extract_features.py \
  --ply /path/to/point_cloud.ply \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth \
  --output outputs/object_features.npz \
  --seed 0
```

The output stores a normalized 768-D object embedding, 64 local 384-D token
features, group centers, sampled Gaussians, and sampling metadata. The exact
input contract and validated scope are specified in [`MODEL_CARD.md`](MODEL_CARD.md).

## Repository structure

```text
assets/             figures used by this README
cfgs/               pretraining and ModelNet transfer configurations
models/             Gaussian-JEPA and downstream model definitions
datasets/           Gaussian loaders and dataset splits
tools/              training, feature extraction, and frozen evaluation
completion_gs/      Gaussian shape completion
segmentation_gs/    ShapeNet-Part segmentation
viz/                visualization and rendering utilities
docs/               setup guides and evaluation protocols
```

## Acknowledgements

This implementation builds on
[ShapeSplat / Gaussian-MAE](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE)
and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). We thank their
authors for releasing code and data. Third-party provenance and license terms
are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License and citation

This repository is released under the
[Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).
Third-party components remain subject to their original terms. The paper link
and BibTeX entry will be added when the public manuscript becomes available.

Please use [`CONTRIBUTING.md`](CONTRIBUTING.md) for bug reports and proposed
extensions.
