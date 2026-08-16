# Robustness to partial Gaussian observations

Frozen encoders are evaluated on all 2,467 ModelNet40-GS test objects. A
complete, independently sampled 1K-Gaussian input forms each gallery embedding.
For each of five observation seeds, queries retain a spatially contiguous subset
of the same object's 64 groups. Both methods receive identical Gaussian inputs
and masks.

| Missing groups | Gaussian-MAE R@1 | Gaussian-JEPA R@1 |
|---:|---:|---:|
| 0% | 92.97 | **93.14** |
| 30% | 60.11 | **70.68** |
| 55% | 19.80 | **39.82** |
| 70% | 6.41 | **19.77** |
| 85% | 1.97 | **6.52** |

Normalized R@1 AUC is 41.82 for Gaussian-MAE and **52.74** for Gaussian-JEPA.
The result measures partial-to-complete instance retrieval; no parameters are
trained during evaluation.

## Reproduce

```bash
python tools/eval_partial_retrieval.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt checkpoints/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/partial_observation

python viz/viz_partial_observation_paper.py \
  --result-dir outputs/partial_observation \
  --output outputs/figures/partial_observation_robustness
```
