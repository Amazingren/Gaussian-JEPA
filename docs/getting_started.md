# Getting started

This guide reproduces the canonical Gaussian-JEPA model and its public
evaluations. All commands are run from the repository root unless stated
otherwise.

## 1. Install

```bash
git clone https://github.com/Amazingren/Gaussian-JEPA.git
cd Gaussian-JEPA
conda env create -f environment.yml
conda activate gaussian_jepa
```

The two CUDA extensions in `environment.yml` are compiled during installation.
Confirm that `nvcc --version` reports CUDA 11.8 and that the target GPU is
visible before starting an experiment.

## 2. Configure data

Download ShapeSplatsV1 for pretraining and ModelNet Splats for classification.
The loaders accept environment variables, so no tracked source file needs to be
edited:

```bash
export SHAPENET55GS_PLY_ROOT=/path/to/shapesplat_ply
export MODELNETGS_PLY_ROOT=/path/to/modelsplat_ply
```

The expected ModelNet layout is:

```text
$MODELNETGS_PLY_ROOT/
  airplane/train/airplane_0001/point_cloud.ply
  airplane/test/airplane_0002/point_cloud.ply
  ...
```

ShapeNet assets are read by the IDs in `datasets/shapenet_split/`. ModelNet
splits are tracked in `datasets/modelnet_split/`. Copy `local.env.example` to
`local.env` if persistent shell settings are convenient; `local.env` is ignored
by Git.

## 3. Download the checkpoint

Download `gaussian_jepa_ep300.pth` from the folder linked in
[`CHECKPOINTS.md`](../CHECKPOINTS.md), then verify it:

```bash
mkdir -p checkpoints
sha256sum checkpoints/gaussian_jepa_ep300.pth
```

The digest must match `CHECKPOINTS.md`. The downstream loader selects the
`JEPA_encoder.*` tensors automatically.

Before configuring a dataset, validate the installation and checkpoint on a
GPU node:

```bash
python tools/smoke_test_release.py \
  --checkpoint checkpoints/gaussian_jepa_ep300.pth
```

This dataset-free test performs strict checkpoint loading and one synthetic
forward/backward step through the CUDA grouping operators.

## 4. Pretrain

```bash
python main.py \
  --config cfgs/pretrain/gaussian_jepa.yaml \
  --exp_name gaussian_jepa \
  --launcher none \
  --seed 0
```

The canonical configuration uses 1,024 Gaussians, 64 groups of 32 neighbors,
four target blocks of sizes `[11, 9, 7, 5]`, total batch size 256, and 300
epochs. Outputs are written under
`experiments/gaussian_jepa/pretrain/gaussian_jepa/`.

## 5. Transfer to ModelNet

The six public configurations cover Full, Linear, and MLP-3 transfer on
ModelNet10 and ModelNet40. For example:

```bash
python main.py \
  --config cfgs/finetune/modelnet10_linear.yaml \
  --finetune_model \
  --ckpts checkpoints/gaussian_jepa_ep300.pth \
  --exp_name modelnet10_linear \
  --seed 0
```

Change only the configuration filename to select another dataset or transfer
protocol. The final accuracy and best checkpoint are recorded in the generated
experiment directory.

## 6. Gaussian-native diagnostics

The resampling and partial-observation evaluators compare frozen Gaussian-JEPA
and Gaussian-MAE encoders using identical sampled primitives, groups, and
masks. Place a compatible Gaussian-MAE checkpoint at
`checkpoints/gaussian_mae_ep300.pth`, or pass its path explicitly.

```bash
python tools/eval_frozen_embeddings.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt checkpoints/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/resampling

python tools/eval_partial_retrieval.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt checkpoints/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/partial_observation
```

`--num-objects 0` evaluates all 2,467 ModelNet40-GS test objects. Use a positive
value such as 200 for a category-balanced pilot.

## 7. Additional tasks

- Shape completion: [`completion_gs/README.md`](../completion_gs/README.md)
- Part segmentation: [`segmentation_gs/README.md`](../segmentation_gs/README.md)
- Visualizations: [`viz/README.md`](../viz/README.md)

Experiment outputs, datasets, and checkpoints are ignored by Git. Keep the
configuration, seed, checkpoint digest, and generated metrics file together
when reporting a new result.
