# ShapeNet-Part segmentation

The segmentation task transfers a Gaussian-pretrained encoder to dense part
prediction. Gaussian features are propagated to the ShapeNet-Part point-label
positions, allowing evaluation with the standard class and instance mIoU
metrics.

## Data

Download the ShapeNet-Part benchmark and corresponding Gaussian assets. Set:

```bash
export PARTANNO_ROOT=/path/to/shapenetcore_partanno_segmentation_benchmark_v0_normal
export PARTSEG_GS_ROOT=/path/to/shapenet_part_gaussians
```

The released `split_to_org_gs_map.json` maps benchmark samples to Gaussian
assets. If the local release uses different object names, regenerate it. The
mapping tool takes paths explicitly and does not modify the dataset:

```bash
python segmentation_gs/prepare_seg_gs.py \
  --partanno-root "$PARTANNO_ROOT" \
  --gs-root "$PARTSEG_GS_ROOT" \
  --output segmentation_gs/split_to_org_gs_map.json
```

## Fine-tune

Run from the repository root:

```bash
python segmentation_gs/main.py \
  --ckpts checkpoints/gaussian_jepa_ep300.pth \
  --partanno_root "$PARTANNO_ROOT" \
  --gs_root "$PARTSEG_GS_ROOT" \
  --pc_to_gs_map segmentation_gs/split_to_org_gs_map.json \
  --log_dir outputs/partseg/gaussian_jepa \
  --attribute '["xyz","opacity","scale","rotation","sh"]' \
  --norm_attribute '["xyz"]' \
  --group_attribute '["xyz"]' \
  --npoint 2048 \
  --num_group 128 \
  --batch_size 16 \
  --epoch 300 \
  --seed 0
```

The checkpoint loader selects `JEPA_encoder.*` automatically. Training logs
report best accuracy, class-average mIoU, and instance-average mIoU; task
checkpoints are written under the selected log directory.

## Released protocol

| Item | Value |
|---|---:|
| Input Gaussians | 2,048 |
| Gaussian groups | 128 |
| Neighbors per group | 32 |
| Input attributes | all 14 |
| Grouping attributes | xyz |
| Batch size | 16 |
| Epochs | 300 |

The reported Gaussian-JEPA result is 84.5 class mIoU and 86.1 instance mIoU.
Qualitative export tools are documented in [`viz/README.md`](../viz/README.md).
