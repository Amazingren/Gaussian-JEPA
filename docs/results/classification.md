# ModelNet classification

Gaussian-JEPA and Gaussian-MAE use 1K Gaussians for both pretraining and
transfer. Overall accuracy is reported in percent.

| Protocol | Gaussian-MAE | Gaussian-JEPA |
|---|---:|---:|
| Full fine-tuning | 94.16 / 92.54 | **94.94 / 92.63** |
| Linear probing | 93.50 / 88.97 | **93.72 / 90.47** |
| MLP-3 probing | 93.39 / 87.72 | **94.16 / 90.27** |

Each cell lists ModelNet10 / ModelNet40. Full fine-tuning updates the encoder and
classification head. Linear and MLP-3 protocols freeze the pretrained encoder
and optimize only the corresponding head.

## Reproduce

Use the six YAML files in `cfgs/finetune/`. For example:

```bash
python main.py \
  --config cfgs/finetune/modelnet40_linear.yaml \
  --finetune_model \
  --ckpts checkpoints/gaussian_jepa_ep300.pth \
  --exp_name modelnet40_linear \
  --seed 0
```

All configurations use 1,024 Gaussians, 64 groups, group size 32, all Gaussian
attributes as encoder input, xyz-only grouping, and `soft_knn=false`.
