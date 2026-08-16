# Representation consistency under Gaussian resampling

For each of the 2,467 ModelNet40-GS test objects, seed 0 forms a reference
embedding and seeds 1--4 form independent 1K-Gaussian queries. Both frozen
encoders receive identical sampled indices and groups. Embeddings concatenate
mean- and max-pooled token features and are L2-normalized.

| Method | Drift down | R@1 (%) up | R@5 (%) up |
|---|---:|---:|---:|
| Gaussian-MAE | 0.1202 | 92.95 | 98.89 |
| Gaussian-JEPA | **0.0850** | **93.00** | **99.36** |

Gaussian-JEPA reduces mean drift by 29.3% and has lower per-object drift on
99.4% of the test set. Retrieval verifies that the more stable representation
retains instance identity.

## Reproduce

```bash
python tools/eval_frozen_embeddings.py \
  --jepa-ckpt checkpoints/gaussian_jepa_ep300.pth \
  --mae-ckpt checkpoints/gaussian_mae_ep300.pth \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --num-objects 0 \
  --output-dir outputs/resampling

python viz/viz_resampling_paper.py \
  --result-dir outputs/resampling \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --output outputs/figures/resampling_consistency
```

The claim is restricted to independently sampled Gaussian subsets of the same
asset; it is not invariance to every render-equivalent Gaussian
reparameterization.
