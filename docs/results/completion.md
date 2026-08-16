# Gaussian shape completion

The completion evaluation asks whether frozen pretrained features support
partial-to-complete Gaussian prediction. Gaussian-JEPA and Gaussian-MAE receive
the same contiguous 512-Gaussian observations and use identical trainable
decoders to predict independently sampled 1K-Gaussian targets.

| Method | CD ↓ | F-score at 1% ↑ | PSNR ↑ | Foreground PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| Gaussian-MAE | 0.0732 | 6.62 | 16.03 | 11.31 | 0.7148 |
| Gaussian-JEPA | **0.0678** | **7.42** | **17.24** | **12.38** | **0.7469** |

CD and F-score are averaged over three independently trained decoders. Render
metrics use the seed-0 decoder at 50% visibility. PSNR values are in dB and
F-score is reported as a percentage. Full task commands are documented in
[`completion_gs/README.md`](../../completion_gs/README.md).
