# ShapeNet-Part segmentation

| Method | Class mIoU (%) | Instance mIoU (%) |
|---|---:|---:|
| Gaussian-MAE | 84.2 | 86.0 |
| Gaussian-JEPA | **84.5** | **86.1** |

Both Gaussian methods use all attributes during pretraining and the same
downstream protocol. The input contains 2,048 Gaussians, with 128 groups of 32
neighbors. The segmentation head predicts labels at the ShapeNet-Part point
positions after Gaussian-to-point feature propagation.

See [`segmentation_gs/README.md`](../../segmentation_gs/README.md) for data
preparation and the direct training command.
