"""Render a fixed Partial | MAE | JEPA | Ground Truth comparison grid."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from completion_gs.data import ShapeNetCompletionDataset, to_render_fields
from completion_gs.train import load_completion
from viz.render_gs import render


def render_normalized(gaussians, stats, device, center, radius, resolution, azimuth, elevation):
    fields = to_render_fields(gaussians, stats)
    return render(
        *fields,
        elevation,
        azimuth,
        2.6,
        resolution,
        40.0,
        [1.0, 1.0, 1.0],
        device,
        center=center,
        radius=radius,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_checkpoint", required=True)
    parser.add_argument("--mae_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--visible_ratio", type=float, default=0.5)
    parser.add_argument("--case_seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--split_root", default=str(ROOT / "datasets/shapenet_split"))
    parser.add_argument(
        "--gs_root",
        default=os.environ.get("SHAPENET55GS_PLY_ROOT", "data/shapesplat_ply"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa, jepa_checkpoint = load_completion(args.jepa_checkpoint, device)
    mae, mae_checkpoint = load_completion(args.mae_checkpoint, device)
    jepa.eval(); mae.eval()
    jepa_args = jepa_checkpoint.get("args", {})
    mae_args = mae_checkpoint.get("args", {})
    sampling = (
        int(jepa_args.get("partial_points", 512)),
        int(jepa_args.get("target_points", 1024)),
    )
    mae_sampling = (
        int(mae_args.get("partial_points", 512)),
        int(mae_args.get("target_points", 1024)),
    )
    if sampling != mae_sampling:
        raise ValueError(f"checkpoint sampling mismatch: JEPA={sampling}, MAE={mae_sampling}")
    dataset = ShapeNetCompletionDataset(
        os.path.join(args.split_root, "test.txt"),
        args.gs_root,
        partial_points=sampling[0], target_points=sampling[1],
        visible_ratios=[args.visible_ratio],
        repeat_seeds=[args.case_seed],
        train=False,
    )
    rows = []
    labels = []
    with torch.no_grad():
        for index in args.indices:
            sample = dataset[index]
            partial = sample["partial"].unsqueeze(0).to(device)
            stats = sample["stats"].to(device)
            target = sample["target"].to(device)
            pred_mae = mae(partial)[0]
            pred_jepa = jepa(partial)[0]
            center = 0.5 * (target[:, :3].amin(0) + target[:, :3].amax(0))
            radius = (target[:, :3] - center).norm(dim=-1).max()
            panels = [partial[0], pred_mae, pred_jepa, target]
            rows.append([
                render_normalized(
                    panel, stats, device, center, radius,
                    args.resolution, args.azimuth, args.elevation,
                )
                for panel in panels
            ])
            labels.append(sample["model_id"])

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"]})
    figure, axes = plt.subplots(len(rows), 4, figsize=(8.0, 2.0 * len(rows)), squeeze=False)
    titles = ["Partial input", "Gaussian-MAE", "Gaussian-JEPA (Ours)", "Ground truth"]
    for row_index, panels in enumerate(rows):
        for column, image in enumerate(panels):
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column], fontsize=9)
        axes[row_index, 0].text(
            0.02, 0.04, labels[row_index], transform=axes[row_index, 0].transAxes,
            fontsize=7, color="black", ha="left", va="bottom",
        )
    figure.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.01, hspace=0.01)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.01)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)
    print(f"saved {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
