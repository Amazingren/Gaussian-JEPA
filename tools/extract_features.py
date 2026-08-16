#!/usr/bin/env python3
"""Extract a frozen Gaussian-JEPA representation from one 3DGS PLY asset.

The script follows the public ModelNet transfer preprocessing: it reads the
14 Gaussian attributes, normalizes xyz (and the physical scale by the same
object radius), samples a fixed 1K budget, forms 64 xyz-based neighborhoods,
and encodes all tokens with the released online encoder.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATTRIBUTES = ["xyz", "opacity", "scale", "rotation", "sh"]


class _Config:
    """Minimal attribute-access configuration used by the public encoder."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def encoder_config() -> _Config:
    transformer = _Config(
        trans_dim=384,
        encoder_dims=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
    )
    return _Config(
        attribute=ATTRIBUTES,
        group_attribute=["xyz"],
        transformer_config=transformer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen Gaussian-JEPA features from one 3DGS PLY."
    )
    parser.add_argument("--ply", type=Path, required=True, help="Input point_cloud.ply")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument("--num-gaussians", type=int, default=1024)
    parser.add_argument("--num-groups", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. The released grouping operators require CUDA.",
    )
    return parser.parse_args()


def normalize_gaussians(data: np.ndarray) -> np.ndarray:
    """Apply the xyz-only downstream normalization used by ModelNet-GS."""
    result = data.astype(np.float32, copy=True)
    xyz = result[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = float(np.sqrt(np.sum(xyz**2, axis=1)).max())
    if not math.isfinite(radius) or radius <= 1e-8:
        raise ValueError(f"invalid object radius: {radius}")
    result[:, :3] = xyz / radius
    result[:, 4:7] /= radius
    return result


def load_asset(path: Path) -> np.ndarray:
    # Keep PLY/HDF5 dependencies out of CLI startup so ``--help`` remains
    # available before the project environment is activated.
    from datasets.ModelNetGaussian import read_gaussian_attribute
    from datasets.io import IO

    if not path.is_file():
        raise FileNotFoundError(path)
    ply = IO.get(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    data = read_gaussian_attribute(ply["vertex"], ATTRIBUTES)
    if data.ndim != 2 or data.shape[1] != 14:
        raise ValueError(f"expected an N x 14 Gaussian array, got {data.shape}")
    return normalize_gaussians(data)


def sample_fixed_budget(data: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        raise ValueError("--num-gaussians must be positive")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(data), count, replace=len(data) < count).astype(np.int64)
    return data[indices], indices


def load_encoder(checkpoint_path: Path, device: torch.device):
    from models.Gaussian_JEPA_ExpMultiScale import _EncoderCore

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("base_model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    prefix = "JEPA_encoder."
    encoder_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError(f"no {prefix!r} tensors found in {checkpoint_path}")

    encoder = _EncoderCore(encoder_config())
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible encoder checkpoint: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return encoder.requires_grad_(False).to(device).eval()


@torch.inference_mode()
def extract(
    points: torch.Tensor,
    encoder,
    group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighborhood, centers = group(points)
    center_position = centers[..., encoder.pos_feature_dim]
    tokens = encoder.encoder(neighborhood)
    tokens = encoder.norm(encoder.blocks(tokens, encoder.pos_embed(center_position)))
    embedding = F.normalize(
        torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1), dim=-1
    )
    return embedding, tokens, centers


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command on a GPU node")
    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must use the .npz extension")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    full_data = load_asset(args.ply)
    sampled, sampled_indices = sample_fixed_budget(
        full_data, args.num_gaussians, args.seed
    )
    points = torch.from_numpy(sampled).unsqueeze(0).to(device=device, dtype=torch.float32)

    encoder = load_encoder(args.checkpoint, device)
    from models.transformer import Group

    group = Group(
        num_group=args.num_groups,
        group_size=args.group_size,
        attribute=["xyz"],
        soft_knn=False,
    ).to(device).eval()
    embedding, tokens, centers = extract(points, encoder, group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embedding=embedding[0].cpu().numpy().astype(np.float32),
        token_features=tokens[0].cpu().numpy().astype(np.float32),
        group_centers=centers[0].cpu().numpy().astype(np.float32),
        sampled_gaussians=sampled.astype(np.float32),
        sampled_indices=sampled_indices,
        source_ply=np.asarray(str(args.ply.resolve())),
        checkpoint=np.asarray(str(args.checkpoint.resolve())),
        seed=np.asarray(args.seed, dtype=np.int64),
    )

    print("Gaussian-JEPA feature extraction: PASS")
    print(f"  source={args.ply} ({len(full_data):,} Gaussians)")
    print(f"  sampled={points.shape[1]:,}, tokens={tuple(tokens.shape)}")
    print(f"  pooled_embedding={tuple(embedding.shape)}, L2={embedding.norm().item():.6f}")
    print(f"  output={args.output}")


if __name__ == "__main__":
    main()
