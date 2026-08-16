#!/usr/bin/env python3
"""Run a minimal GPU smoke test for the public Gaussian-JEPA release.

The test builds the canonical pretraining model, optionally loads a released
checkpoint strictly, and runs one forward/backward pass on a synthetic batch
of valid 14-D Gaussian primitives. It intentionally requires no dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model construction, checkpoint loading, and CUDA training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cfgs/pretrain/gaussian_jepa.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional released pretraining checkpoint to load strictly.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def synthetic_gaussians(batch_size: int, num_points: int, device: torch.device) -> torch.Tensor:
    """Construct [xyz, opacity, scale, quaternion, SH] Gaussian parameters."""
    points = torch.zeros(batch_size, num_points, 14, device=device)
    points[..., :3] = torch.randn(batch_size, num_points, 3, device=device)
    points[..., 3] = torch.sigmoid(torch.randn(batch_size, num_points, device=device))
    points[..., 4:7] = 0.02 + 0.08 * torch.rand(batch_size, num_points, 3, device=device)

    quaternion = torch.randn(batch_size, num_points, 4, device=device)
    points[..., 7:11] = torch.nn.functional.normalize(quaternion, dim=-1)
    points[..., 11:14] = torch.randn(batch_size, num_points, 3, device=device) * 0.1
    return points


def load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("base_model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this test on a GPU compute node")

    # Delay CUDA-extension imports until after argument parsing, so --help and
    # static inspection also work on login nodes without a CUDA runtime.
    from models import build_model_from_cfg
    from utils.config import cfg_from_yaml_file

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = cfg_from_yaml_file(str(args.config))
    model = build_model_from_cfg(config.model)
    if args.checkpoint is not None:
        load_checkpoint_strict(model, args.checkpoint)
        checkpoint_status = f"strict load passed: {args.checkpoint}"
    else:
        checkpoint_status = "checkpoint load skipped"

    device = torch.device("cuda")
    model = model.to(device).train()
    points = synthetic_gaussians(args.batch_size, int(config.npoints), device)

    model.zero_grad(set_to_none=True)
    losses = model(points)
    if not isinstance(losses, dict) or not losses:
        raise TypeError(f"expected a non-empty loss dictionary, got {type(losses)!r}")
    for name, value in losses.items():
        if value.ndim != 0 or not torch.isfinite(value):
            raise RuntimeError(f"invalid loss {name}: {value}")

    total_loss = sum(losses.values())
    total_loss.backward()
    trainable_gradients = sum(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_gradients == 0:
        raise RuntimeError("backward pass produced no trainable gradients")

    loss_text = ", ".join(f"{name}={value.item():.6f}" for name, value in losses.items())
    peak_memory = (
        f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
        if device.type == "cuda"
        else "n/a"
    )
    print("Gaussian-JEPA release smoke test: PASS")
    print(f"  torch={torch.__version__}, cuda_build={torch.version.cuda}, device={device}")
    print(f"  {checkpoint_status}")
    print(f"  input={tuple(points.shape)}, losses: {loss_text}")
    print(f"  parameters_with_grad={trainable_gradients}, peak_memory={peak_memory}")


if __name__ == "__main__":
    main()
