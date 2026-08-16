#!/usr/bin/env python3
"""Compute embedding-space-normalized drift from saved resampling features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(path: Path, chunk_size: int) -> tuple[dict[str, float], np.ndarray]:
    data = np.load(path)
    embeddings = data["embeddings"].astype(np.float32)
    gallery = embeddings[:, 0]
    relative_by_seed = []
    same_distances = []
    nonmatching_distances = []

    for seed_index in range(1, embeddings.shape[1]):
        queries = embeddings[:, seed_index]
        seed_relative = np.empty(len(gallery), dtype=np.float64)
        for start in range(0, len(gallery), chunk_size):
            end = min(start + chunk_size, len(gallery))
            similarities = queries[start:end] @ gallery.T
            distances = np.sqrt(np.maximum(2.0 - 2.0 * similarities, 0.0))
            local = np.arange(end - start)
            matching = distances[local, np.arange(start, end)]
            nonmatching = (
                distances.sum(axis=1) - matching
            ) / float(len(gallery) - 1)
            seed_relative[start:end] = matching / np.maximum(
                nonmatching, 1e-12
            )
            same_distances.append(matching)
            nonmatching_distances.append(nonmatching)
        relative_by_seed.append(seed_relative)

    relative = np.stack(relative_by_seed, axis=1)
    per_object = relative.mean(axis=1)
    summary = {
        "relative_drift_mean": float(per_object.mean()),
        "relative_drift_std": float(per_object.std(ddof=1)),
        "same_object_distance_mean": float(
            np.concatenate(same_distances).mean()
        ),
        "nonmatching_distance_mean": float(
            np.concatenate(nonmatching_distances).mean()
        ),
        "objects": int(len(gallery)),
        "queries_per_object": int(embeddings.shape[1] - 1),
    }
    return summary, per_object


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    methods = {}
    per_object = {}
    for method in ("jepa", "mae"):
        methods[method], per_object[method] = summarize(
            args.result_dir / f"{method}_embeddings.npz",
            args.chunk_size,
        )

    jepa_mean = methods["jepa"]["relative_drift_mean"]
    mae_mean = methods["mae"]["relative_drift_mean"]
    result = {
        "methods": methods,
        "comparison": {
            "relative_reduction_vs_mae": float(
                (mae_mean - jepa_mean) / mae_mean
            ),
            "object_win_fraction": float(
                np.mean(per_object["jepa"] < per_object["mae"])
            ),
        },
    }

    output = args.output or args.result_dir / "relative_drift.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved relative drift to {output}")


if __name__ == "__main__":
    main()
