# Visualization utilities

The release contains the scripts needed to reproduce Gaussian-native figures,
completion renderings, and ShapeNet-Part qualitative panels. Generated files
are written under `outputs/` by default.

## Gaussian resampling

After running `tools/eval_frozen_embeddings.py`:

```bash
python viz/viz_resampling_paper.py \
  --result-dir outputs/resampling \
  --gs-root "$MODELNETGS_PLY_ROOT" \
  --output outputs/figures/resampling_consistency
```

The script generates a compact empirical CDF of relative embedding drift and
true Gaussian-splat examples. The legacy multi-panel diagnostic generator is
also available as `viz_resampling_consistency.py`.

## Partial observations

```bash
python viz/viz_partial_observation_paper.py \
  --result-dir outputs/partial_observation \
  --output outputs/figures/partial_observation_robustness
```

## Render a Gaussian asset

```bash
python viz/render_gs.py \
  --ply /path/to/point_cloud.ply \
  --out outputs/render \
  --azim 45 135 225 315 \
  --elev 20 \
  --res 800
```

## Part segmentation

The qualitative pipeline has three stages: choose deterministic ShapeNet-Part
cases, export predictions from each checkpoint, and render matched panels.

```bash
python viz/export_partseg_qualitative.py prepare \
  --data-root "$PARTANNO_ROOT" \
  --output-root outputs/partseg_qualitative

python viz/export_partseg_predictions.py \
  --method gaussian_jepa \
  --checkpoint /path/to/partseg_checkpoint.pth \
  --data-root "$PARTANNO_ROOT" \
  --gs-root "$PARTSEG_GS_ROOT" \
  --output-root outputs/partseg_qualitative

python viz/export_partseg_qualitative.py render \
  --output-root outputs/partseg_qualitative
```

The same exporter supports Point-MAE, Point-JEPA, and Gaussian-MAE checkpoints.
Set `POINT_MAE_ROOT`, `POINT_JEPA_ROOT`, or `GAUSSIAN_MAE_ROOT` when those
repositories are not adjacent to Gaussian-JEPA. Every method is evaluated on
the exact point indices recorded in `cases_manifest.json`.

Completion visualizations are documented in
[`completion_gs/README.md`](../completion_gs/README.md).
