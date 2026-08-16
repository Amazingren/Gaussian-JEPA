#!/usr/bin/env python3
"""Paper-ready visualizations for frozen-embedding resampling consistency.

This script consumes outputs from ``tools/eval_frozen_embeddings.py`` and does
not run either encoder. It produces:

1. a quantitative paired comparison over the complete test split;
2. representative objects under all Gaussian sampling seeds, accompanied by
   their frozen-embedding similarity to the seed-0 reference;
3. cross-sampling retrieval examples where Gaussian-JEPA retrieves the correct
   object and Gaussian-MAE does not.

Representative consistency examples are selected deterministically as the
objects closest to the category-median drift improvement. Retrieval examples
are selected from distinct categories by the largest MAE-vs-JEPA rank gap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "outputs/resampling"
DEFAULT_GS_ROOT = Path(os.environ.get("MODELNETGS_PLY_ROOT", "data/modelsplat_ply"))
SH_C0 = 0.28209479177387814

JEPA_COLOR = "#087E8B"
MAE_COLOR = "#D95F59"
NEUTRAL_COLOR = "#51606F"
GRID_COLOR = "#D9DEE5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS_ROOT)
    parser.add_argument("--subset", default="test")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["airplane", "chair", "sofa", "table"],
        help="Categories used for representative consistency rows.",
    )
    parser.add_argument("--num-retrieval-cases", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_outputs(result_dir: Path):
    paths = {
        "jepa": result_dir / "jepa_embeddings.npz",
        "mae": result_dir / "mae_embeddings.npz",
        "records": result_dir / "per_object.csv",
        "metrics": result_dir / "metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    arrays = {name: np.load(paths[name]) for name in ("jepa", "mae")}
    object_ids = arrays["jepa"]["object_ids"].astype(str)
    categories = arrays["jepa"]["categories"].astype(str)
    sample_seeds = arrays["jepa"]["sample_seeds"].astype(int)
    for key in ("object_ids", "categories", "sample_seeds", "input_indices"):
        if not np.array_equal(arrays["jepa"][key], arrays["mae"][key]):
            raise ValueError(f"JEPA/MAE mismatch in shared field: {key}")
    if sample_seeds[0] != 0:
        raise ValueError("sample seed 0 must be the reference gallery")

    records = pd.read_csv(paths["records"])
    metrics = json.loads(paths["metrics"].read_text())
    return arrays, object_ids, categories, sample_seeds, records, metrics


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def object_level_table(records: pd.DataFrame) -> pd.DataFrame:
    return (
        records.groupby(["object_id", "category", "method"], as_index=False)[
            ["cosine_to_seed0", "drift_to_seed0", "retrieval_rank"]
        ]
        .mean()
        .pivot(index=["object_id", "category"], columns="method")
    )


def make_quantitative_figure(
    records: pd.DataFrame, metrics: dict, output: Path, dpi: int
) -> Dict[str, float]:
    obj = object_level_table(records)
    mae_drift = obj["drift_to_seed0"]["mae"].to_numpy()
    jepa_drift = obj["drift_to_seed0"]["jepa"].to_numpy()
    improvement = mae_drift - jepa_drift

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.15))

    ax = axes[0]
    limit = float(max(mae_drift.max(), jepa_drift.max()) * 1.04)
    ax.scatter(
        mae_drift,
        jepa_drift,
        s=8,
        alpha=0.24,
        color=JEPA_COLOR,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, limit], [0, limit], linestyle="--", color=NEUTRAL_COLOR, linewidth=1)
    ax.set(xlim=(0, limit), ylim=(0, limit))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gaussian-MAE drift")
    ax.set_ylabel("Gaussian-JEPA drift")
    ax.set_title("Paired object-level drift")
    lower_fraction = float(np.mean(jepa_drift < mae_drift))
    ax.text(
        0.04,
        0.95,
        f"{lower_fraction:.1%} below $y=x$",
        transform=ax.transAxes,
        va="top",
        color=JEPA_COLOR,
        fontweight="bold",
    )
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    ax = axes[1]
    for method, color, label in (
        ("jepa", JEPA_COLOR, "Gaussian-JEPA"),
        ("mae", MAE_COLOR, "Gaussian-MAE"),
    ):
        x, y = empirical_cdf(
            records.loc[records["method"] == method, "drift_to_seed0"].to_numpy()
        )
        ax.plot(x, y, color=color, linewidth=2, label=label)
    ax.set_xlabel("Embedding drift to seed 0")
    ax.set_ylabel("Cumulative fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("All 9,868 paired queries")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    ax = axes[2]
    metric_names = ["r_at_1", "r_at_5", "mrr"]
    labels = ["R@1", "R@5", "MRR"]
    jepa_values = [metrics["methods"]["jepa"]["aggregate"][k] for k in metric_names]
    mae_values = [metrics["methods"]["mae"]["aggregate"][k] for k in metric_names]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, jepa_values, width, color=JEPA_COLOR, label="Gaussian-JEPA")
    ax.bar(x + width / 2, mae_values, width, color=MAE_COLOR, label="Gaussian-MAE")
    ax.set_ylim(0.90, 1.005)
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Cross-sampling retrieval")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False
    )
    for positions, values in ((x - width / 2, jepa_values), (x + width / 2, mae_values)):
        for xpos, value in zip(positions, values):
            ax.text(xpos, value + 0.002, f"{100 * value:.1f}", ha="center", fontsize=7)

    fig.suptitle(
        "Frozen representation consistency under Gaussian resampling",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output, dpi)

    return {
        "object_fraction_jepa_lower_drift": lower_fraction,
        "mean_jepa_drift": float(jepa_drift.mean()),
        "mean_mae_drift": float(mae_drift.mean()),
        "relative_drift_reduction": float(improvement.mean() / mae_drift.mean()),
    }


def read_gaussian_view(ply_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(
        np.float32
    )
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = float(np.sqrt(np.sum(xyz**2, axis=1)).max())
    xyz /= max(radius, 1e-8)
    fdc = np.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=1
    ).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float32)))
    return xyz, rgb, opacity


def configure_cloud_axis(ax, xyz: np.ndarray, title: str = "") -> None:
    ax.view_init(elev=18, azim=-55)
    ax.set_proj_type("ortho")
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    center = (lower + upper) / 2.0
    half_extent = max(float((upper - lower).max()) * 0.55, 1e-3)
    ax.set(
        xlim=(center[0] - half_extent, center[0] + half_extent),
        ylim=(center[1] - half_extent, center[1] + half_extent),
        zlim=(center[2] - half_extent, center[2] + half_extent),
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=-2)


def plot_sample_cloud(
    ax, xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray, indices: np.ndarray, title: str
) -> None:
    sampled_xyz = xyz[indices]
    sampled_rgb = rgb[indices]
    sampled_alpha = np.clip(opacity[indices], 0.18, 0.9)
    colors = np.concatenate([sampled_rgb, sampled_alpha[:, None]], axis=1)
    ax.scatter(
        sampled_xyz[:, 0],
        sampled_xyz[:, 1],
        sampled_xyz[:, 2],
        c=colors,
        s=2.4,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    # Keep identical camera framing across all resamplings of this object.
    configure_cloud_axis(ax, xyz, title)


def representative_objects(
    records: pd.DataFrame, categories: Sequence[str]
) -> List[Dict[str, object]]:
    obj = object_level_table(records)
    rows: List[Dict[str, object]] = []
    for category in categories:
        try:
            subset = obj.xs(category, level="category")
        except KeyError as error:
            raise ValueError(f"unknown category: {category}") from error
        gain = subset["drift_to_seed0"]["mae"] - subset["drift_to_seed0"]["jepa"]
        median = float(gain.median())
        object_id = str((gain - median).abs().idxmin())
        rows.append(
            {
                "object_id": object_id,
                "category": category,
                "drift_reduction": float(gain.loc[object_id]),
                "selection": "closest to category-median drift reduction",
            }
        )
    return rows


def make_sampling_panel(
    arrays,
    object_ids: np.ndarray,
    sample_seeds: np.ndarray,
    records: pd.DataFrame,
    examples: Sequence[Dict[str, object]],
    gs_root: Path,
    subset: str,
    output: Path,
    dpi: int,
) -> None:
    id_to_index = {object_id: index for index, object_id in enumerate(object_ids)}
    fig = plt.figure(figsize=(13.2, 2.45 * len(examples)))
    grid = fig.add_gridspec(
        len(examples),
        len(sample_seeds) + 1,
        width_ratios=[1] * len(sample_seeds) + [1.45],
        wspace=0.02,
        hspace=0.08,
    )

    for row, example in enumerate(examples):
        object_id = str(example["object_id"])
        category = str(example["category"])
        object_index = id_to_index[object_id]
        ply_path = gs_root / category / subset / object_id / "point_cloud.ply"
        xyz, rgb, opacity = read_gaussian_view(ply_path)
        indices = arrays["jepa"]["input_indices"][object_index]
        for seed_index, seed in enumerate(sample_seeds):
            ax = fig.add_subplot(grid[row, seed_index], projection="3d")
            plot_sample_cloud(
                ax,
                xyz,
                rgb,
                opacity,
                indices[seed_index],
                f"Seed {seed}" if row == 0 else "",
            )
            if seed_index == 0:
                ax.text2D(
                    -0.06,
                    0.5,
                    object_id,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                    fontweight="bold",
                )

        ax = fig.add_subplot(grid[row, -1])
        obj_records = records[records["object_id"] == object_id]
        for method, color, label in (
            ("jepa", JEPA_COLOR, "Gaussian-JEPA"),
            ("mae", MAE_COLOR, "Gaussian-MAE"),
        ):
            method_records = obj_records[obj_records["method"] == method].sort_values(
                "sample_seed"
            )
            ax.plot(
                method_records["sample_seed"],
                method_records["cosine_to_seed0"],
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=color,
                label=label,
            )
        ax.set_xticks(sample_seeds[1:])
        ax.set_ylim(0.975, 1.0005)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.8)
        if row == 0:
            ax.set_title("Similarity to seed 0")
            ax.legend(frameon=False, loc="lower left")
        if row == len(examples) - 1:
            ax.set_xlabel("Sampling seed")
        ax.set_ylabel("Cosine")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)

    fig.suptitle(
        "The same object under five Gaussian resamplings",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    save_figure(fig, output, dpi)


def retrieval_candidates(
    arrays,
    object_ids: np.ndarray,
    categories: np.ndarray,
    sample_seeds: np.ndarray,
) -> List[Dict[str, object]]:
    gallery = {method: arrays[method]["embeddings"][:, 0] for method in ("jepa", "mae")}
    candidates: List[Dict[str, object]] = []
    for seed_index, seed in enumerate(sample_seeds[1:], start=1):
        similarities = {
            method: arrays[method]["embeddings"][:, seed_index] @ gallery[method].T
            for method in ("jepa", "mae")
        }
        order = {method: np.argsort(-value, axis=1) for method, value in similarities.items()}
        for query_index, object_id in enumerate(object_ids):
            jepa_top = int(order["jepa"][query_index, 0])
            mae_top = int(order["mae"][query_index, 0])
            if jepa_top != query_index or mae_top == query_index:
                continue
            mae_rank = int(np.flatnonzero(order["mae"][query_index] == query_index)[0] + 1)
            candidates.append(
                {
                    "query_index": query_index,
                    "object_id": str(object_id),
                    "category": str(categories[query_index]),
                    "sample_seed": int(seed),
                    "seed_index": seed_index,
                    "jepa_top_index": jepa_top,
                    "mae_top_index": mae_top,
                    "mae_rank": mae_rank,
                    "rank_gap": mae_rank - 1,
                    "mae_top_object_id": str(object_ids[mae_top]),
                    "mae_top_category": str(categories[mae_top]),
                }
            )
    return candidates


def choose_diverse_retrieval_cases(
    candidates: Sequence[Dict[str, object]], count: int
) -> List[Dict[str, object]]:
    ordered = sorted(candidates, key=lambda row: (-int(row["rank_gap"]), str(row["object_id"])))
    chosen: List[Dict[str, object]] = []
    used_categories = set()
    used_objects = set()
    for row in ordered:
        if row["category"] in used_categories or row["object_id"] in used_objects:
            continue
        chosen.append(dict(row))
        used_categories.add(row["category"])
        used_objects.add(row["object_id"])
        if len(chosen) == count:
            return chosen
    for row in ordered:
        if row["object_id"] in used_objects:
            continue
        chosen.append(dict(row))
        used_objects.add(row["object_id"])
        if len(chosen) == count:
            return chosen
    return chosen


def make_retrieval_panel(
    arrays,
    object_ids: np.ndarray,
    categories: np.ndarray,
    cases: Sequence[Dict[str, object]],
    gs_root: Path,
    subset: str,
    output: Path,
    dpi: int,
) -> None:
    if not cases:
        raise RuntimeError("no JEPA-success/MAE-failure retrieval cases found")
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def cloud(index: int):
        if index not in cache:
            path = (
                gs_root
                / str(categories[index])
                / subset
                / str(object_ids[index])
                / "point_cloud.ply"
            )
            cache[index] = read_gaussian_view(path)
        return cache[index]

    fig = plt.figure(figsize=(10.4, 2.35 * len(cases)))
    grid = fig.add_gridspec(len(cases), 4, wspace=0.02, hspace=0.04)
    titles = ["Query resampling", "JEPA top-1", "MAE top-1", "Ground-truth gallery"]
    for row, case in enumerate(cases):
        query_index = int(case["query_index"])
        mae_top = int(case["mae_top_index"])
        seed_index = int(case["seed_index"])
        entries = [
            (query_index, seed_index),
            (query_index, 0),
            (mae_top, 0),
            (query_index, 0),
        ]
        for column, (object_index, input_seed_index) in enumerate(entries):
            ax = fig.add_subplot(grid[row, column], projection="3d")
            xyz, rgb, opacity = cloud(object_index)
            indices = arrays["jepa"]["input_indices"][object_index, input_seed_index]
            title = titles[column] if row == 0 else ""
            plot_sample_cloud(ax, xyz, rgb, opacity, indices, title)
            if column == 0:
                ax.text2D(
                    -0.02,
                    0.5,
                    f"{case['category']}\nseed {case['sample_seed']}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                    fontweight="bold",
                )
            if column == 2:
                ax.text2D(
                    0.5,
                    0.04,
                    f"wrong: {case['mae_top_category']}\nGT rank {case['mae_rank']}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=MAE_COLOR,
                )
            if column == 1:
                ax.text2D(
                    0.5,
                    0.04,
                    "correct",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=JEPA_COLOR,
                    fontweight="bold",
                )
    fig.suptitle(
        "Cross-sampling retrieval: illustrative JEPA-success / MAE-failure cases",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    save_figure(fig, output, dpi)


def main() -> None:
    args = parse_args()
    set_style()
    output_dir = args.output_dir or args.result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays, object_ids, categories, sample_seeds, records, metrics = load_outputs(
        args.result_dir
    )
    summary = make_quantitative_figure(
        records, metrics, output_dir / "resampling_quantitative", args.dpi
    )

    examples = representative_objects(records, args.categories)
    make_sampling_panel(
        arrays,
        object_ids,
        sample_seeds,
        records,
        examples,
        args.gs_root,
        args.subset,
        output_dir / "resampling_same_object",
        args.dpi,
    )

    all_retrieval_cases = retrieval_candidates(
        arrays, object_ids, categories, sample_seeds
    )
    retrieval_cases = choose_diverse_retrieval_cases(
        all_retrieval_cases, args.num_retrieval_cases
    )
    make_retrieval_panel(
        arrays,
        object_ids,
        categories,
        retrieval_cases,
        args.gs_root,
        args.subset,
        output_dir / "resampling_retrieval_examples",
        args.dpi,
    )

    report = {
        "protocol": metrics["protocol"],
        "summary": summary,
        "representative_consistency_examples": examples,
        "retrieval_selection": "distinct categories, descending MAE-vs-JEPA rank gap",
        "retrieval_cases": retrieval_cases,
        "available_jepa_success_mae_failure_queries": len(all_retrieval_cases),
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"Saved {output_dir / 'visualization_manifest.json'}")


if __name__ == "__main__":
    main()
