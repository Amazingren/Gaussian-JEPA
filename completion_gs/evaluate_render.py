"""Multi-view render metrics for a trained Gaussian completion decoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from completion_gs.data import ShapeNetCompletionDataset, to_render_fields
from completion_gs.train import load_completion
from viz.render_gs import render


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visible_ratio", type=float, default=0.5)
    parser.add_argument("--case_seed", type=int, default=0)
    parser.add_argument("--views", type=float, nargs="+", default=[45.0, 135.0, 225.0, 315.0])
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_objects", type=int, default=0)
    parser.add_argument("--split_root", default=str(ROOT / "datasets/shapenet_split"))
    parser.add_argument(
        "--gs_root",
        default=os.environ.get("SHAPENET55GS_PLY_ROOT", "data/shapesplat_ply"),
    )
    return parser.parse_args()


def render_normalized(gaussians, stats, device, center, radius, resolution, azimuth, elevation):
    return render(
        *to_render_fields(gaussians, stats),
        elevation, azimuth, 2.6, resolution, 40.0, [1.0, 1.0, 1.0], device,
        center=center, radius=radius,
    ).astype(np.float32) / 255.0


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0)
    gt = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
    mu_pred = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_gt = F.avg_pool2d(gt, 11, stride=1, padding=5)
    var_pred = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_pred.square()
    var_gt = F.avg_pool2d(gt * gt, 11, 1, 5) - mu_gt.square()
    covariance = F.avg_pool2d(pred * gt, 11, 1, 5) - mu_pred * mu_gt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_pred * mu_gt + c1) * (2 * covariance + c2)) / (
        (mu_pred.square() + mu_gt.square() + c1) * (var_pred + var_gt + c2)
    ).clamp_min(1e-8)
    return float(score.mean())


def image_metrics(prediction: np.ndarray, target: np.ndarray):
    mse = float(np.mean((prediction - target) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    foreground = np.logical_or(
        np.any(target < 0.99, axis=-1), np.any(prediction < 0.99, axis=-1)
    )
    foreground_mse = float(np.mean((prediction[foreground] - target[foreground]) ** 2))
    foreground_psnr = -10.0 * math.log10(max(foreground_mse, 1e-12))
    return psnr, foreground_psnr, ssim(prediction, target)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_completion(args.checkpoint, device)
    model.eval()
    training_args = checkpoint.get("args", {})
    partial_points = int(training_args.get("partial_points", 512))
    target_points = int(training_args.get("target_points", 1024))
    dataset = ShapeNetCompletionDataset(
        os.path.join(args.split_root, "test.txt"), args.gs_root,
        partial_points=partial_points, target_points=target_points,
        visible_ratios=[args.visible_ratio], repeat_seeds=[args.case_seed], train=False,
    )
    count = len(dataset) if args.num_objects <= 0 else min(args.num_objects, len(dataset))
    records = []
    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            partial = sample["partial"].unsqueeze(0).to(device)
            target = sample["target"].to(device)
            stats = sample["stats"].to(device)
            prediction = model(partial)[0]
            center = 0.5 * (target[:, :3].amin(0) + target[:, :3].amax(0))
            radius = (target[:, :3] - center).norm(dim=-1).max()
            for azimuth in args.views:
                pred_image = render_normalized(
                    prediction, stats, device, center, radius,
                    args.resolution, azimuth, args.elevation,
                )
                target_image = render_normalized(
                    target, stats, device, center, radius,
                    args.resolution, azimuth, args.elevation,
                )
                psnr, foreground_psnr, score = image_metrics(pred_image, target_image)
                records.append({
                    "model_id": sample["model_id"], "azimuth": azimuth,
                    "psnr": psnr, "foreground_psnr": foreground_psnr, "ssim": score,
                })
            if (index + 1) % 25 == 0:
                print(f"[{index + 1}/{count}]", flush=True)

    summary = {
        "method": checkpoint.get("method", "unknown"), "objects": count,
        "views_per_object": len(args.views), "visible_ratio": args.visible_ratio,
        "partial_points": partial_points,
        "target_points": target_points,
        "psnr": float(np.mean([row["psnr"] for row in records])),
        "foreground_psnr": float(np.mean([row["foreground_psnr"] for row in records])),
        "ssim": float(np.mean([row["ssim"] for row in records])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(output.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
