#!/usr/bin/env python3
"""Generate quantitative and true-splat resampling figures for the paper."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VIZ_DIR = ROOT / "viz"
if str(VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(VIZ_DIR))

from viz_resampling_consistency import (  # noqa: E402
    GRID_COLOR,
    JEPA_COLOR,
    MAE_COLOR,
)
from render_gs import load_gaussians, render  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "outputs/resampling",
    )
    parser.add_argument(
        "--gs-root",
        type=Path,
        default=Path(os.environ.get("MODELNETGS_PLY_ROOT", "data/modelsplat_ply")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/figures/resampling_consistency",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()



def set_paper_style() -> None:
    """Use a compact Times-compatible style for AAAI single-column figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "Liberation Serif"],
            "font.size": 7,
            "mathtext.fontset": "stix",
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.72,
            "lines.antialiased": True,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print("Saved", stem.with_suffix(".pdf"))
    print("Saved", stem.with_suffix(".png"))


def plot_quantitative(result_dir: Path, output: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(3.25, 1.55))

    for method, color, label in (
        ("jepa", MAE_COLOR, "Gaussian-JEPA (Ours)"),
        ("mae", JEPA_COLOR, "Gaussian-MAE"),
    ):
        embeddings = np.load(result_dir / f"{method}_embeddings.npz")[
            "embeddings"
        ].astype(np.float64)
        gallery = embeddings[:, 0]
        per_query = []
        for sample_index in range(1, embeddings.shape[1]):
            query = embeddings[:, sample_index]
            similarity = np.clip(query @ gallery.T, -1.0, 1.0)
            distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
            paired = distance[np.arange(len(gallery)), np.arange(len(gallery))]
            nonmatching = (distance.sum(axis=1) - paired) / (len(gallery) - 1)
            per_query.append(paired / nonmatching)
        values = np.stack(per_query, axis=1).mean(axis=1)
        values.sort()
        cumulative = np.arange(1, len(values) + 1) / len(values)
        window = np.hanning(41)
        window /= window.sum()
        padded = np.pad(values, len(window) // 2, mode="edge")
        smooth_values = np.convolve(padded, window, mode="valid")
        axis.plot(smooth_values, cumulative, color=color, label=label)
        axis.axvline(
            values.mean(),
            color=color,
            linewidth=0.58,
            linestyle="--",
            alpha=0.9,
        )

    axis.set_xlabel(r"Relative embedding drift $\downarrow$")
    axis.set_ylabel("Fraction of test objects")
    axis.set_xlim(0.14, 0.58)
    axis.set_ylim(0.0, 1.01)
    axis.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        handlelength=1.45,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.28, top=0.98)
    save_figure(fig, output, dpi)






def crop_row_to_shared_foreground(
    images: list[np.ndarray],
    padding: int = 8,
) -> list[np.ndarray]:
    """Apply one tight foreground crop to every image in a row."""
    masks = [np.any(image < 248, axis=2) for image in images]
    union = np.logical_or.reduce(masks)
    y, x = np.nonzero(union)
    if len(x) == 0:
        return images
    x0 = max(int(x.min()) - padding, 0)
    x1 = min(int(x.max()) + padding + 1, images[0].shape[1])
    y0 = max(int(y.min()) - padding, 0)
    y1 = min(int(y.max()) + padding + 1, images[0].shape[0])
    return [image[y0:y1, x0:x1] for image in images]


def plot_resampling_gallery(
    result_dir: Path,
    gs_root: Path,
    output: Path,
    dpi: int,
) -> None:
    """Render two ModelNet40-GS assets and three evaluated 1K samples."""
    examples = [
        ("car_0209", 15.0, -30.0, -45.0),
        ("guitar_0156", 15.0, 0.0, -180.0),
    ]
    archive = np.load(result_dir / "jepa_embeddings.npz")
    object_ids = archive["object_ids"].astype(str)
    input_indices_all = archive["input_indices"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for object_id, elev, azim, rotation_degrees in examples:
        matches = np.flatnonzero(object_ids == object_id)
        if len(matches) != 1:
            raise ValueError(f"expected one match for {object_id}, found {len(matches)}")
        object_index = int(matches[0])
        category = object_id.rsplit("_", 1)[0]
        ply_path = gs_root / category / "test" / object_id / "point_cloud.ply"
        xyz, scale, quat, opacity, color = load_gaussians(str(ply_path), device)

        lower = xyz.amin(dim=0)
        upper = xyz.amax(dim=0)
        center = 0.5 * (lower + upper)
        radius = (xyz - center).norm(dim=-1).max()
        sampled = input_indices_all[object_index]
        panel_indices = [
            torch.arange(xyz.shape[0], device=device),
            torch.as_tensor(sampled[0], dtype=torch.long, device=device),
            torch.as_tensor(sampled[2], dtype=torch.long, device=device),
            torch.as_tensor(sampled[4], dtype=torch.long, device=device),
        ]

        images = []
        for indices in panel_indices:
            images.append(
                render(
                    xyz[indices],
                    scale[indices],
                    quat[indices],
                    opacity[indices],
                    color[indices],
                    elev=elev,
                    azim=azim,
                    dist_mul=3.5,
                    res=440,
                    fov=40.0,
                    bg=[1.0, 1.0, 1.0],
                    device=device,
                    center=center,
                    radius=radius,
                )
            )
        if rotation_degrees:
            images = [
                np.asarray(
                    Image.fromarray(image).rotate(
                        rotation_degrees,
                        resample=Image.Resampling.BICUBIC,
                        expand=True,
                        fillcolor=(255, 255, 255),
                    )
                )
                for image in images
            ]
        rows.append(crop_row_to_shared_foreground(images))

    fig, axes = plt.subplots(2, 4, figsize=(5.0, 2.15))
    for row, images in enumerate(rows):
        for column, image in enumerate(images):
            axes[row, column].imshow(image, interpolation="lanczos")
            axes[row, column].set_axis_off()

    labels = [
        "(a) Full GS",
        "(b) Resampling 1 (1K)",
        "(c) Resampling 2 (1K)",
        "(d) Resampling 3 (1K)",
    ]
    for column, label in enumerate(labels):
        fig.text(
            (column + 0.5) / 4.0,
            0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.12,
        top=0.998,
        wspace=0.0,
        hspace=0.005,
    )
    save_figure(fig, output, dpi)


def main() -> None:
    args = parse_args()
    set_paper_style()
    plot_quantitative(args.result_dir, args.output, args.dpi)

    plot_resampling_gallery(
        args.result_dir,
        args.gs_root,
        args.output.with_name("resampling_inputs"),
        args.dpi,
    )


if __name__ == "__main__":
    main()
