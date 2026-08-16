#!/usr/bin/env python3
"""Frozen partial-to-complete retrieval for Gaussian-JEPA vs Gaussian-MAE.

The complete seed-0 embedding of each ModelNet40-GS object forms the gallery.
Independent seed-1..5 Gaussian subsets form queries. For every query, an
oriented spatial cap of group tokens is retained at several visibility levels;
both encoders receive exactly the same sampled Gaussians, groups, directions,
and visible-token masks. No parameters are trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_frozen_embeddings import (
    ROOT,
    Group,
    category_from_id,
    encode_shared_batch,
    fps_gs,
    load_encoders,
    load_full_gaussians,
    load_split,
    per_object_rng,
    pool_tokens,
    sample_candidates,
    select_objects,
)


DEFAULT_JEPA_CKPT = (
    ROOT / "checkpoints/gaussian_jepa_ep300.pth"
)
DEFAULT_MAE_CKPT = ROOT / "checkpoints/gaussian_mae_ep300.pth"
DEFAULT_SPLIT = ROOT / "datasets/modelnet_split/modelnet40_test.txt"
DEFAULT_GS_ROOT = Path(os.environ.get("MODELNETGS_PLY_ROOT", "data/modelsplat_ply"))


def observation_direction(object_id: str, seed: int) -> np.ndarray:
    rng = per_object_rng(object_id, 100_000 + int(seed))
    direction = rng.normal(size=3).astype(np.float32)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"invalid observation direction: {object_id}, seed={seed}")
    return direction / norm


def visible_group_counts(num_groups: int, missing_ratios: Sequence[float]) -> List[int]:
    return [max(1, min(num_groups, int(round(num_groups * (1.0 - ratio))))) for ratio in missing_ratios]


@torch.inference_mode()
def encode_partial_batch(
    candidate_batch: np.ndarray,
    original_index_batch: np.ndarray,
    object_ids: Sequence[str],
    observation_seeds: Sequence[int],
    encoders: Dict[str, torch.nn.Module],
    group: Group,
    npoints: int,
    missing_ratios: Sequence[float],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = torch.from_numpy(candidate_batch).to(device=device, dtype=torch.float32)
    fps_index = fps_gs(candidates, npoints, attribute=["xyz"], return_idx=True).long()
    gather_index = fps_index.unsqueeze(-1).expand(-1, -1, candidates.size(-1))
    points = torch.gather(candidates, 1, gather_index).contiguous()
    neighborhood, center = group(points)
    center_xyz = center[..., :3]

    direction_numpy = np.stack(
        [observation_direction(object_id, seed) for object_id, seed in zip(object_ids, observation_seeds)]
    )
    directions = torch.from_numpy(direction_numpy).to(device=device, dtype=torch.float32)
    projection = torch.sum(center_xyz * directions[:, None, :], dim=-1)

    counts = visible_group_counts(center_xyz.size(1), missing_ratios)
    masks: List[torch.Tensor] = []
    for count in counts:
        indices = torch.topk(projection, k=count, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(projection, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        masks.append(mask)

    jepa = encoders["jepa"]
    mae = encoders["mae"]
    raw_tokens = {
        "jepa": jepa.encoder(neighborhood),
        "mae": mae.encoder(neighborhood),
    }
    outputs: Dict[str, List[torch.Tensor]] = {"jepa": [], "mae": []}
    batch_size = center_xyz.size(0)

    for mask, count in zip(masks, counts):
        center_vis = center_xyz[mask].reshape(batch_size, count, 3)
        for method, model in (("jepa", jepa), ("mae", mae)):
            tokens_vis = raw_tokens[method][mask].reshape(batch_size, count, -1)
            encoded = model.norm(model.blocks(tokens_vis, model.pos_embed(center_vis)))
            outputs[method].append(pool_tokens(encoded))

    embeddings = {
        method: torch.stack(values, dim=1).cpu().numpy().astype(np.float32)
        for method, values in outputs.items()
    }
    fps_numpy = fps_index.cpu().numpy()
    final_indices = np.take_along_axis(original_index_batch, fps_numpy, axis=1).astype(np.int32)
    visible_masks = torch.stack(masks, dim=1).cpu().numpy().astype(np.uint8)
    return (
        embeddings,
        final_indices,
        center_xyz.cpu().numpy().astype(np.float32),
        direction_numpy,
        visible_masks,
    )


def retrieval_metrics(
    gallery: np.ndarray,
    queries: np.ndarray,
    observation_seeds: Sequence[int],
    missing_ratios: Sequence[float],
) -> Dict[str, object]:
    # queries: objects x seeds x ratios x dim
    per_ratio: Dict[str, Dict[str, object]] = {}
    for ratio_index, ratio in enumerate(missing_ratios):
        per_seed: Dict[str, Dict[str, float]] = {}
        all_cosine, all_drift, all_ranks = [], [], []
        for seed_index, seed in enumerate(observation_seeds):
            query = queries[:, seed_index, ratio_index]
            cosine = np.sum(query * gallery, axis=1)
            drift = np.linalg.norm(query - gallery, axis=1)
            order = np.argsort(-(query @ gallery.T), axis=1)
            ranks = np.argmax(order == np.arange(len(query))[:, None], axis=1) + 1
            per_seed[str(seed)] = metric_row(cosine, drift, ranks)
            all_cosine.append(cosine)
            all_drift.append(drift)
            all_ranks.append(ranks)
        cosine = np.concatenate(all_cosine)
        drift = np.concatenate(all_drift)
        ranks = np.concatenate(all_ranks)
        aggregate = metric_row(cosine, drift, ranks)
        aggregate["num_queries"] = int(len(ranks))
        per_ratio[f"{ratio:.2f}"] = {"aggregate": aggregate, "per_seed": per_seed}
    return per_ratio


def metric_row(cosine: np.ndarray, drift: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(ddof=1)),
        "drift_mean": float(drift.mean()),
        "drift_std": float(drift.std(ddof=1)),
        "r_at_1": float(np.mean(ranks <= 1)),
        "r_at_5": float(np.mean(ranks <= 5)),
        "r_at_10": float(np.mean(ranks <= 10)),
        "mean_rank": float(ranks.mean()),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def write_per_object_csv(
    path: Path,
    object_ids: Sequence[str],
    seeds: Sequence[int],
    ratios: Sequence[float],
    counts: Sequence[int],
    gallery: Dict[str, np.ndarray],
    queries: Dict[str, np.ndarray],
) -> None:
    fields = [
        "method", "object_id", "category", "observation_seed", "missing_ratio",
        "visible_groups", "cosine_to_complete", "drift_to_complete", "retrieval_rank",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in ("jepa", "mae"):
            complete = gallery[method]
            for seed_index, seed in enumerate(seeds):
                for ratio_index, (ratio, count) in enumerate(zip(ratios, counts)):
                    query = queries[method][:, seed_index, ratio_index]
                    cosine = np.sum(query * complete, axis=1)
                    drift = np.linalg.norm(query - complete, axis=1)
                    order = np.argsort(-(query @ complete.T), axis=1)
                    ranks = np.argmax(order == np.arange(len(query))[:, None], axis=1) + 1
                    for index, object_id in enumerate(object_ids):
                        writer.writerow(
                            {
                                "method": method,
                                "object_id": object_id,
                                "category": category_from_id(object_id),
                                "observation_seed": seed,
                                "missing_ratio": f"{ratio:.2f}",
                                "visible_groups": count,
                                "cosine_to_complete": f"{cosine[index]:.8f}",
                                "drift_to_complete": f"{drift[index]:.8f}",
                                "retrieval_rank": int(ranks[index]),
                            }
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-ckpt", type=Path, default=DEFAULT_JEPA_CKPT)
    parser.add_argument("--mae-ckpt", type=Path, default=DEFAULT_MAE_CKPT)
    parser.add_argument("--jepa-prefix", choices=["JEPA_encoder.", "teacher_encoder."], default="JEPA_encoder.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS_ROOT)
    parser.add_argument("--subset", default="test")
    parser.add_argument("--num-objects", type=int, default=200)
    parser.add_argument("--object-seed", type=int, default=2027)
    parser.add_argument("--observation-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--missing-ratios", type=float, nargs="+", default=[0.0, 0.30, 0.55, 0.70, 0.85])
    parser.add_argument("--candidate-count", type=int, default=8192)
    parser.add_argument("--npoints", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/partial_observation",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.jepa_ckpt, args.mae_ckpt, args.split_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.gs_root.is_dir():
        raise NotADirectoryError(args.gs_root)
    if not args.observation_seeds or len(set(args.observation_seeds)) != len(args.observation_seeds):
        raise ValueError("observation seeds must be non-empty and unique")
    if any(ratio < 0 or ratio >= 1 for ratio in args.missing_ratios):
        raise ValueError("missing ratios must satisfy 0 <= ratio < 1")
    if sorted(args.missing_ratios) != args.missing_ratios or len(set(args.missing_ratios)) != len(args.missing_ratios):
        raise ValueError("missing ratios must be unique and sorted")
    if args.candidate_count < args.npoints or args.batch_size <= 0:
        raise ValueError("invalid candidate-count/npoints/batch-size")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    all_ids = load_split(args.split_file)
    object_ids = select_objects(all_ids, args.num_objects, args.object_seed)
    categories = [category_from_id(object_id) for object_id in object_ids]
    counts = visible_group_counts(64, args.missing_ratios)
    print(
        f"Selected {len(object_ids)}/{len(all_ids)} objects, {len(set(categories))} categories; "
        f"seeds={args.observation_seeds}; missing={args.missing_ratios}; visible_groups={counts}"
    )

    encoders = load_encoders(args.jepa_ckpt, args.mae_ckpt, args.jepa_prefix, device)
    group = Group(num_group=64, group_size=32, attribute=["xyz"], soft_knn=False).to(device).eval()
    object_count, seed_count, ratio_count = len(object_ids), len(args.observation_seeds), len(args.missing_ratios)
    gallery = {method: np.empty((object_count, 768), np.float32) for method in ("jepa", "mae")}
    queries = {
        method: np.empty((object_count, seed_count, ratio_count, 768), np.float32)
        for method in ("jepa", "mae")
    }
    input_indices = np.empty((object_count, seed_count + 1, args.npoints), np.int32)
    query_centers = np.empty((object_count, seed_count, 64, 3), np.float32)
    directions = np.empty((object_count, seed_count, 3), np.float32)
    visibility = np.empty((object_count, seed_count, ratio_count, 64), np.uint8)

    gallery_data, gallery_indices, gallery_records = [], [], []
    query_data, query_indices, query_records = [], [], []
    processed_gallery = processed_query = 0

    def flush_gallery() -> None:
        nonlocal processed_gallery
        if not gallery_data:
            return
        encoded, final_indices = encode_shared_batch(
            np.stack(gallery_data).astype(np.float32),
            np.stack(gallery_indices).astype(np.int64),
            encoders, group, args.npoints, device,
        )
        for batch_index, object_index in enumerate(gallery_records):
            for method in gallery:
                gallery[method][object_index] = encoded[method][batch_index]
            input_indices[object_index, 0] = final_indices[batch_index]
        processed_gallery += len(gallery_records)
        print(f"Gallery {processed_gallery}/{object_count}", flush=True)
        gallery_data.clear(); gallery_indices.clear(); gallery_records.clear()

    def flush_queries() -> None:
        nonlocal processed_query
        if not query_data:
            return
        batch_ids = [object_ids[object_index] for object_index, _ in query_records]
        batch_seeds = [args.observation_seeds[seed_index] for _, seed_index in query_records]
        encoded, final_indices, centers, batch_directions, masks = encode_partial_batch(
            np.stack(query_data).astype(np.float32),
            np.stack(query_indices).astype(np.int64),
            batch_ids, batch_seeds, encoders, group, args.npoints,
            args.missing_ratios, device,
        )
        for batch_index, (object_index, seed_index) in enumerate(query_records):
            for method in queries:
                queries[method][object_index, seed_index] = encoded[method][batch_index]
            input_indices[object_index, seed_index + 1] = final_indices[batch_index]
            query_centers[object_index, seed_index] = centers[batch_index]
            directions[object_index, seed_index] = batch_directions[batch_index]
            visibility[object_index, seed_index] = masks[batch_index]
        processed_query += len(query_records)
        print(f"Queries {processed_query}/{object_count * seed_count}", flush=True)
        query_data.clear(); query_indices.clear(); query_records.clear()

    for object_index, object_id in enumerate(object_ids):
        category = categories[object_index]
        ply_path = args.gs_root / category / args.subset / object_id / "point_cloud.ply"
        if not ply_path.is_file():
            raise FileNotFoundError(ply_path)
        full_data = load_full_gaussians(ply_path)

        candidates, indices = sample_candidates(full_data, object_id, 0, args.candidate_count)
        gallery_data.append(candidates); gallery_indices.append(indices); gallery_records.append(object_index)
        if len(gallery_data) >= args.batch_size:
            flush_gallery()

        for seed_index, seed in enumerate(args.observation_seeds):
            candidates, indices = sample_candidates(full_data, object_id, seed, args.candidate_count)
            query_data.append(candidates); query_indices.append(indices); query_records.append((object_index, seed_index))
            if len(query_data) >= args.batch_size:
                flush_queries()
    flush_gallery(); flush_queries()

    metrics = {
        "protocol": {
            "dataset": "ModelNet40-GS",
            "split_file": str(args.split_file),
            "split_objects": len(all_ids),
            "evaluated_objects": object_count,
            "categories": len(set(categories)),
            "gallery_sample_seed": 0,
            "observation_seeds": args.observation_seeds,
            "missing_ratios": args.missing_ratios,
            "visible_groups": counts,
            "masking": "oriented spatial cap: retain groups with largest center projection",
            "candidate_count": args.candidate_count,
            "npoints": args.npoints,
            "grouping": "xyz FPS, 64 groups x 32 neighbors",
            "pooling": "L2-normalized concat(mean(tokens), max(tokens))",
            "jepa_checkpoint": str(args.jepa_ckpt),
            "jepa_prefix": args.jepa_prefix,
            "mae_checkpoint": str(args.mae_ckpt),
        },
        "methods": {
            method: retrieval_metrics(gallery[method], queries[method], args.observation_seeds, args.missing_ratios)
            for method in ("jepa", "mae")
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in ("jepa", "mae"):
        np.savez_compressed(
            args.output_dir / f"{method}_embeddings.npz",
            object_ids=np.asarray(object_ids), categories=np.asarray(categories),
            observation_seeds=np.asarray(args.observation_seeds, np.int32),
            missing_ratios=np.asarray(args.missing_ratios, np.float32),
            gallery=gallery[method], queries=queries[method],
        )
    np.savez_compressed(
        args.output_dir / "partial_inputs.npz",
        object_ids=np.asarray(object_ids), observation_seeds=np.asarray(args.observation_seeds, np.int32),
        missing_ratios=np.asarray(args.missing_ratios, np.float32), input_indices=input_indices,
        query_center_xyz=query_centers, observation_directions=directions, visible_group_masks=visibility,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_per_object_csv(
        args.output_dir / "per_object.csv", object_ids, args.observation_seeds,
        args.missing_ratios, counts, gallery, queries,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"object_ids": object_ids, "categories": categories}, indent=2) + "\n"
    )
    print(json.dumps(metrics["methods"], indent=2, sort_keys=True))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
