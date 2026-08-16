# Gaussian shape completion

This task measures whether a frozen pretrained encoder supports
partial-to-complete Gaussian prediction. The encoder receives only a spatially
contiguous partial observation; an independently sampled complete target is
used for supervision. Gaussian-JEPA and Gaussian-MAE use identical decoder
architectures and data splits.

## Protocol

- Partial input: 512 Gaussians from a contiguous visible region.
- Complete target: an independent 1,024-Gaussian sample of the same asset.
- Backbone: frozen throughout completion training.
- Decoder: learned queries; no missing-region coordinates are exposed.
- Selection: the ShapeNet55-GS training split is divided into train and
  validation subsets, and the official test split is evaluated once.
- Metrics: Chamfer distance, F-score, PSNR, foreground PSNR, and SSIM.

This directory is independent of pretraining and does not modify the released
encoder checkpoint.

## Train Gaussian-JEPA completion

Run from the repository root:

```bash
python completion_gs/train.py \
  --encoder_checkpoint checkpoints/gaussian_jepa_ep300.pth \
  --encoder_prefix JEPA_encoder. \
  --method Gaussian-JEPA \
  --output_dir outputs/completion/gaussian_jepa \
  --gs_root "$SHAPENET55GS_PLY_ROOT" \
  --seed 0
```

For the matched Gaussian-MAE baseline, change the checkpoint, use
`--encoder_prefix MAE_encoder.`, and write to a separate output directory.
Training produces `best.pth`, `last.pth`, `history.jsonl`,
`test_metrics.json`, and `test_per_case.csv`.

Before a full run, append `--smoke` to validate the data, frozen encoder, and
decoder on a small subset.

## Render-space evaluation

```bash
python completion_gs/evaluate_render.py \
  --checkpoint outputs/completion/gaussian_jepa/best.pth \
  --output outputs/completion/gaussian_jepa/render_metrics.json \
  --gs_root "$SHAPENET55GS_PLY_ROOT" \
  --visible_ratio 0.5
```

The evaluator renders four views per object and reports both whole-image and
foreground PSNR so that a white background cannot hide incomplete geometry.

## Matched qualitative comparison

After training both decoders:

```bash
python completion_gs/visualize_completion_paper.py \
  --jepa_checkpoint outputs/completion/gaussian_jepa/best.pth \
  --mae_checkpoint outputs/completion/gaussian_mae/best.pth \
  --gs_root "$SHAPENET55GS_PLY_ROOT" \
  --output outputs/completion/qualitative.pdf \
  --num_examples 8
```

Each row uses a shared camera and crop. Predictions retain the quantitative
1K output budget; the full source asset is shown only as a visual reference.
