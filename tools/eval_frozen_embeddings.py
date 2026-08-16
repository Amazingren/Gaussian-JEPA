#!/usr/bin/env python3
"""Paired frozen-embedding evaluation for Gaussian-JEPA and Gaussian-MAE.

The first protocol implemented here is sampling consistency. For each ModelNet
Gaussian asset, several deterministic 8K candidate sets are sampled from the
full PLY and reduced to 1K with xyz FPS, matching the current downstream input
pipeline. JEPA and MAE then receive exactly the same grouped inputs.

No parameters are trained. The primary object representation is the L2-normalized
concatenation of mean- and max-pooled token features (384 + 384 dimensions).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.ModelNetGaussian import read_gaussian_attribute  # noqa: E402
from datasets.io import IO  # noqa: E402
from models.Gaussian_JEPA_ExpMultiScale import _EncoderCore  # noqa: E402
from models.Gaussian_MAE import MaskTransformer  # noqa: E402
from models.transformer import Group  # noqa: E402
from utils.misc import fps_gs  # noqa: E402


ATTRIBUTES = ["xyz", "opacity", "scale", "rotation", "sh"]


class _NS:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _transformer_cfg(mask_ratio: float) -> _NS:
    return _NS(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dims=384,
        decoder_depth=4,
        decoder_num_heads=6,
        pred_dim=192,
        mask_ratio=mask_ratio,
        mask_type="block",
    )


def _jepa_cfg() -> _NS:
    return _NS(
        attribute=ATTRIBUTES,
        group_attribute=["xyz"],
        soft_knn=False,
        transformer_config=_transformer_cfg(0.55),
        group_size=32,
        num_group=64,
    )


def _mae_cfg() -> _NS:
    return _NS(
        attribute=ATTRIBUTES,
        group_attribute=["xyz"],
        norm_attribute=["xyz"],
        soft_knn=False,
        transformer_config=_transformer_cfg(0.60),
        group_size=32,
        num_group=64,
    )


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if "base_model" not in checkpoint:
        raise KeyError(f"checkpoint has no 'base_model': {path}")
    return {
        key.replace("module.", ""): value
        for key, value in checkpoint["base_model"].items()
    }


def _load_prefixed_module(
    module: torch.nn.Module, checkpoint: Path, prefix: str, device: torch.device
) -> torch.nn.Module:
    base = _checkpoint_state(checkpoint)
    state = {
        key[len(prefix) :]: value
        for key, value in base.items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError(f"no keys with prefix {prefix!r} in {checkpoint}")
    incompatible = module.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible {prefix} state from {checkpoint}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    module.requires_grad_(False)
    return module.to(device).eval()


def load_encoders(
    jepa_checkpoint: Path,
    mae_checkpoint: Path,
    jepa_prefix: str,
    device: torch.device,
) -> Dict[str, torch.nn.Module]:
    return {
        "jepa": _load_prefixed_module(
            _EncoderCore(_jepa_cfg()), jepa_checkpoint, jepa_prefix, device
        ),
        "mae": _load_prefixed_module(
            MaskTransformer(_mae_cfg()), mae_checkpoint, "MAE_encoder.", device
        ),
    }


def category_from_id(object_id: str) -> str:
    return object_id.rsplit("_", 1)[0]


def load_split(split_file: Path) -> List[str]:
    object_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"duplicate object IDs in {split_file}")
    return object_ids


def select_objects(
    object_ids: Sequence[str], num_objects: int, object_seed: int
) -> List[str]:
    if num_objects <= 0 or num_objects >= len(object_ids):
        return list(object_ids)

    by_category: Dict[str, List[str]] = defaultdict(list)
    for object_id in object_ids:
        by_category[category_from_id(object_id)].append(object_id)

    categories = sorted(by_category)
    if num_objects < len(categories):
        raise ValueError(
            f"num_objects={num_objects} cannot cover all {len(categories)} categories"
        )

    base_quota, remainder = divmod(num_objects, len(categories))
    rng = np.random.RandomState(object_seed)
    selected: List[str] = []
    for category_index, category in enumerate(categories):
        items = np.asarray(sorted(by_category[category]), dtype=object)
        quota = base_quota + int(category_index < remainder)
        if quota > len(items):
            raise ValueError(
                f"category {category} has {len(items)} objects, needs {quota}"
            )
        chosen = rng.choice(items, quota, replace=False)
        selected.extend(str(item) for item in chosen)
    return selected


def normalize_for_downstream(data: np.ndarray) -> np.ndarray:
    """Match ModelNetGaussian.pc_norm_gs(..., norm_attribute=['xyz'])."""
    result = data.astype(np.float32, copy=True)
    xyz = result[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = float(np.sqrt(np.sum(xyz**2, axis=1)).max())
    if not math.isfinite(radius) or radius <= 1e-8:
        raise ValueError(f"invalid xyz normalization radius: {radius}")
    result[:, :3] = xyz / radius
    result[:, 4:7] /= radius
    return result


def load_full_gaussians(ply_path: Path) -> np.ndarray:
    gaussian = IO.get(str(ply_path))
    data = read_gaussian_attribute(gaussian["vertex"], ATTRIBUTES)
    if data.ndim != 2 or data.shape[1] != 14:
        raise ValueError(f"expected Nx14 Gaussian array, got {data.shape}: {ply_path}")
    return normalize_for_downstream(data)


def per_object_rng(object_id: str, sample_seed: int) -> np.random.RandomState:
    object_crc = zlib.crc32(object_id.encode("utf-8")) & 0xFFFFFFFF
    sequence = np.random.SeedSequence([int(sample_seed), int(object_crc)])
    seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return np.random.RandomState(seed)


def sample_candidates(
    full_data: np.ndarray, object_id: str, sample_seed: int, candidate_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = per_object_rng(object_id, sample_seed)
    # This deliberately matches ModelNetGaussian.__getitem__: replacement is always enabled.
    indices = rng.choice(len(full_data), candidate_count, replace=True).astype(np.int64)
    return full_data[indices], indices


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    mean_feature = tokens.mean(dim=1)
    max_feature = tokens.max(dim=1).values
    return F.normalize(torch.cat([mean_feature, max_feature], dim=-1), dim=-1)


@torch.inference_mode()
def encode_shared_batch(
    candidate_batch: np.ndarray,
    original_index_batch: np.ndarray,
    encoders: Dict[str, torch.nn.Module],
    group: Group,
    npoints: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    candidates = torch.from_numpy(candidate_batch).to(device=device, dtype=torch.float32)
    fps_index = fps_gs(
        candidates, npoints, attribute=["xyz"], return_idx=True
    ).long()
    gather_index = fps_index.unsqueeze(-1).expand(-1, -1, candidates.size(-1))
    points = torch.gather(candidates, dim=1, index=gather_index).contiguous()

    neighborhood, center = group(points)
    center_xyz = center[..., :3]

    jepa = encoders["jepa"]
    jepa_tokens = jepa.encoder(neighborhood)
    jepa_tokens = jepa.norm(
        jepa.blocks(jepa_tokens, jepa.pos_embed(center_xyz))
    )

    mae = encoders["mae"]
    mae_tokens, mae_mask = mae(neighborhood, center_xyz, noaug=True)
    if bool(mae_mask.any()):
        raise RuntimeError("Gaussian-MAE noaug=True unexpectedly masked tokens")

    embeddings = {
        "jepa": pool_tokens(jepa_tokens).cpu().numpy().astype(np.float32),
        "mae": pool_tokens(mae_tokens).cpu().numpy().astype(np.float32),
    }
    fps_numpy = fps_index.cpu().numpy()
    final_original_indices = np.take_along_axis(
        original_index_batch, fps_numpy, axis=1
    ).astype(np.int32)
    return embeddings, final_original_indices


def retrieval_metrics(embeddings: np.ndarray, seeds: Sequence[int]) -> Dict[str, object]:
    """Use seed 0 as gallery and every other seed as a same-instance query."""
    if embeddings.ndim != 3:
        raise ValueError(f"expected [objects,seeds,dim], got {embeddings.shape}")
    if len(seeds) < 2:
        raise ValueError("at least two sample seeds are required")

    gallery = embeddings[:, 0]
    per_seed: Dict[str, Dict[str, float]] = {}
    all_cosine: List[np.ndarray] = []
    all_drift: List[np.ndarray] = []
    all_ranks: List[np.ndarray] = []

    for seed_index, seed in enumerate(seeds[1:], start=1):
        query = embeddings[:, seed_index]
        paired_cosine = np.sum(query * gallery, axis=1)
        paired_drift = np.linalg.norm(query - gallery, axis=1)
        similarities = query @ gallery.T
        order = np.argsort(-similarities, axis=1)
        ranks = np.argmax(order == np.arange(len(query))[:, None], axis=1) + 1

        per_seed[str(seed)] = {
            "cosine_mean": float(paired_cosine.mean()),
            "cosine_std": float(paired_cosine.std(ddof=1)),
            "drift_mean": float(paired_drift.mean()),
            "drift_std": float(paired_drift.std(ddof=1)),
            "r_at_1": float(np.mean(ranks <= 1)),
            "r_at_5": float(np.mean(ranks <= 5)),
            "mean_rank": float(ranks.mean()),
            "mrr": float(np.mean(1.0 / ranks)),
        }
        all_cosine.append(paired_cosine)
        all_drift.append(paired_drift)
        all_ranks.append(ranks)

    cosine = np.concatenate(all_cosine)
    drift = np.concatenate(all_drift)
    ranks = np.concatenate(all_ranks)
    aggregate = {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(ddof=1)),
        "drift_mean": float(drift.mean()),
        "drift_std": float(drift.std(ddof=1)),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_5": float(np.mean(ranks <= 5)),
        "mean_rank": float(ranks.mean()),
        "mrr": float(np.mean(1.0 / ranks)),
        "num_queries": int(len(ranks)),
    }
    return {"aggregate": aggregate, "per_seed": per_seed}


def write_per_object_csv(
    path: Path,
    object_ids: Sequence[str],
    seeds: Sequence[int],
    embeddings: Dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "method",
        "object_id",
        "category",
        "sample_seed",
        "cosine_to_seed0",
        "drift_to_seed0",
        "retrieval_rank",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, method_embeddings in embeddings.items():
            gallery = method_embeddings[:, 0]
            for seed_index, seed in enumerate(seeds[1:], start=1):
                query = method_embeddings[:, seed_index]
                similarities = query @ gallery.T
                order = np.argsort(-similarities, axis=1)
                ranks = np.argmax(
                    order == np.arange(len(object_ids))[:, None], axis=1
                ) + 1
                cosine = np.sum(query * gallery, axis=1)
                drift = np.linalg.norm(query - gallery, axis=1)
                for object_index, object_id in enumerate(object_ids):
                    writer.writerow(
                        {
                            "method": method,
                            "object_id": object_id,
                            "category": category_from_id(object_id),
                            "sample_seed": seed,
                            "cosine_to_seed0": f"{cosine[object_index]:.8f}",
                            "drift_to_seed0": f"{drift[object_index]:.8f}",
                            "retrieval_rank": int(ranks[object_index]),
                        }
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jepa-ckpt",
        type=Path,
        default=ROOT / "checkpoints/gaussian_jepa_ep300.pth",
    )
    parser.add_argument(
        "--mae-ckpt",
        type=Path,
        default=ROOT / "checkpoints/gaussian_mae_ep300.pth",
    )
    parser.add_argument(
        "--jepa-prefix",
        default="JEPA_encoder.",
        choices=["JEPA_encoder.", "teacher_encoder."],
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "datasets/modelnet_split/modelnet40_test.txt",
    )
    parser.add_argument(
        "--gs-root",
        type=Path,
        default=Path(os.environ.get("MODELNETGS_PLY_ROOT", "data/modelsplat_ply")),
    )
    parser.add_argument("--subset", default="test")
    parser.add_argument("--num-objects", type=int, default=200)
    parser.add_argument("--object-seed", type=int, default=2027)
    parser.add_argument(
        "--sample-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    parser.add_argument("--candidate-count", type=int, default=8192)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/resampling",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in [args.jepa_ckpt, args.mae_ckpt, args.split_file]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.gs_root.is_dir():
        raise NotADirectoryError(args.gs_root)
    if not args.sample_seeds or args.sample_seeds[0] != 0:
        raise ValueError("sample seed 0 must be first and serves as the gallery")
    if len(args.sample_seeds) != len(set(args.sample_seeds)):
        raise ValueError("sample seeds must be unique")
    if args.candidate_count < args.npoints:
        raise ValueError("candidate-count must be >= npoints")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    all_object_ids = load_split(args.split_file)
    object_ids = select_objects(all_object_ids, args.num_objects, args.object_seed)
    categories = [category_from_id(object_id) for object_id in object_ids]
    print(
        f"Selected {len(object_ids)}/{len(all_object_ids)} objects, "
        f"{len(set(categories))} categories, seeds={args.sample_seeds}"
    )

    encoders = load_encoders(
        args.jepa_ckpt, args.mae_ckpt, args.jepa_prefix, device
    )
    group = Group(
        num_group=64, group_size=32, attribute=["xyz"], soft_knn=False
    ).to(device).eval()

    total_records = len(object_ids) * len(args.sample_seeds)
    flat_embeddings: Dict[str, List[np.ndarray]] = {"jepa": [], "mae": []}
    flat_final_indices: List[np.ndarray] = []
    pending_data: List[np.ndarray] = []
    pending_indices: List[np.ndarray] = []
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not pending_data:
            return
        data_batch = np.stack(pending_data).astype(np.float32, copy=False)
        index_batch = np.stack(pending_indices).astype(np.int64, copy=False)
        batch_embeddings, final_indices = encode_shared_batch(
            data_batch,
            index_batch,
            encoders,
            group,
            args.npoints,
            device,
        )
        for method in flat_embeddings:
            flat_embeddings[method].append(batch_embeddings[method])
        flat_final_indices.append(final_indices)
        processed += len(pending_data)
        print(f"Encoded {processed}/{total_records}", flush=True)
        pending_data.clear()
        pending_indices.clear()

    for object_id in object_ids:
        category = category_from_id(object_id)
        ply_path = (
            args.gs_root / category / args.subset / object_id / "point_cloud.ply"
        )
        if not ply_path.is_file():
            raise FileNotFoundError(ply_path)
        full_data = load_full_gaussians(ply_path)
        for sample_seed in args.sample_seeds:
            candidates, original_indices = sample_candidates(
                full_data, object_id, sample_seed, args.candidate_count
            )
            pending_data.append(candidates)
            pending_indices.append(original_indices)
            if len(pending_data) >= args.batch_size:
                flush()
    flush()

    object_count = len(object_ids)
    seed_count = len(args.sample_seeds)
    embeddings = {
        method: np.concatenate(chunks, axis=0).reshape(object_count, seed_count, -1)
        for method, chunks in flat_embeddings.items()
    }
    final_indices = np.concatenate(flat_final_indices, axis=0).reshape(
        object_count, seed_count, args.npoints
    )

    metrics = {
        "protocol": {
            "dataset": "ModelNet40-GS",
            "split_file": str(args.split_file),
            "split_objects": len(all_object_ids),
            "evaluated_objects": object_count,
            "categories": len(set(categories)),
            "object_seed": args.object_seed,
            "sample_seeds": args.sample_seeds,
            "candidate_count": args.candidate_count,
            "npoints": args.npoints,
            "grouping": "xyz FPS, 64 groups x 32 neighbors",
            "pooling": "L2-normalized concat(mean(tokens), max(tokens))",
            "jepa_checkpoint": str(args.jepa_ckpt),
            "jepa_prefix": args.jepa_prefix,
            "mae_checkpoint": str(args.mae_ckpt),
        },
        "methods": {
            method: retrieval_metrics(method_embeddings, args.sample_seeds)
            for method, method_embeddings in embeddings.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method, method_embeddings in embeddings.items():
        np.savez_compressed(
            args.output_dir / f"{method}_embeddings.npz",
            object_ids=np.asarray(object_ids),
            categories=np.asarray(categories),
            sample_seeds=np.asarray(args.sample_seeds, dtype=np.int32),
            embeddings=method_embeddings,
            input_indices=final_indices,
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    write_per_object_csv(
        args.output_dir / "per_object.csv",
        object_ids,
        args.sample_seeds,
        embeddings,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "object_ids": object_ids,
                "categories": categories,
                "sample_seeds": args.sample_seeds,
            },
            indent=2,
        )
        + "\n"
    )

    print(json.dumps(metrics["methods"], indent=2, sort_keys=True))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
