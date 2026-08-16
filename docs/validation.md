# Release validation

The public code was checked against the reference 300-epoch checkpoint and
a real ShapeSplats-format PLY asset. This page records the interface contract,
not a new benchmark result.

## Checkpoint

- File: `gaussian_jepa_ep300.pth`
- SHA-256: `42e753944cc8e0e763df4bf1c59f465a1f368b5a53c9f87493ec0c07f3eee5bc`
- Payload: 377 tensors under `checkpoint["base_model"]`
- Online encoder prefix: `JEPA_encoder.`

## GPU checks

The following checks passed on an NVIDIA H200 NVL:

1. CUDA grouping extensions imported successfully.
2. The reference checkpoint loaded into the public model implementation.
3. A synthetic pretraining batch produced finite losses and completed one
   backward pass.
4. A standard 14-attribute PLY with 15,613 Gaussians was normalized and sampled
   to the reference 1,024-Gaussian budget.
5. Feature extraction produced 64 local tokens of 384 dimensions and a
   normalized 768-dimensional mean/max-pooled representation.

The dataset-free model test is available in
[`tools/smoke_test_release.py`](../tools/smoke_test_release.py). The real-asset
interface is exercised by
[`tools/extract_features.py`](../tools/extract_features.py). Paths to private
datasets and cluster logs are deliberately not part of the release.

## Static checks

All released Python sources pass syntax compilation, all YAML configurations
parse successfully, and the documentation has no unresolved local links. These
checks do not replace full downstream training; task protocols and expected
metrics are listed under [`docs/results`](results/README.md).
