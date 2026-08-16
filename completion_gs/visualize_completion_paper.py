"""Create paper-quality Partial | MAE | JEPA | Full-GS comparisons.

The qualitative reference is the complete source asset rather than a random
1K target sample. Predictions remain the unchanged 1K outputs used by the
quantitative completion protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from completion_gs.data import (
    ShapeNetCompletionDataset,
    normalize_eall,
    to_render_fields,
)
from completion_gs.train import load_completion
from datasets.ShapeNet55Gaussian import read_gaussian_attribute
from datasets.io import IO
from viz.render_gs_alpha import render_alpha


PANEL_NAMES = ("partial", "gaussian_mae", "gaussian_jepa", "full_gs")
PANEL_TITLES = ("Partial input", "Gaussian-MAE", "Gaussian-JEPA (Ours)", "Full GS")
PANEL_FILES = (
    "01_partial_input.png",
    "02_gaussian_mae.png",
    "03_gaussian_jepa_ours.png",
    "04_complete_target_1kg.png",
    "05_full_gs_original.png",
)
TAXONOMY_NAMES = {
    "02773838": "bag",
    "02958343": "car",
    "03001627": "chair",
    "04225987": "skateboard",
    "03636649": "lamp",
    "03624134": "knife",
    "02828884": "bench",
    "02801938": "basket",
    "04090263": "rifle",
    "02871439": "bookshelf",
    "03467517": "guitar",
    "03211117": "display",
    "04530566": "vessel",
    "04256520": "sofa",
    "02924116": "bus",
    "03928116": "piano",
    "03337140": "file_cabinet",
    "02691156": "airplane",
    "04401088": "telephone",
    "02880940": "bowl",
    "02933112": "cabinet",
    "03691459": "loudspeaker",
    "03642806": "laptop",
    "03790512": "motorcycle",
    "02818832": "bed",
    "04468005": "train",
    "03046257": "clock",
    "04379243": "table",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_checkpoint", required=True)
    parser.add_argument("--mae_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel_root")
    parser.add_argument("--panels_only", action="store_true")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--indices", type=int, nargs="+")
    identity.add_argument("--model_ids", nargs="+")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--visible_ratio", type=float, default=0.5)
    parser.add_argument("--case_seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument("--azimuths", type=float, nargs="+", default=[135.0])
    parser.add_argument("--object_azimuths", type=float, nargs="+")
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--distance", type=float, default=2.6)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--max_splat_radius", type=float, default=32.0)
    parser.add_argument("--alpha_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument(
        "--crop_padding",
        type=float,
        default=0.04,
        help="fractional padding around the shared foreground box for each row",
    )
    parser.add_argument("--no_shared_crop", action="store_true")
    parser.add_argument("--no_geometry_overlay", action="store_true")
    parser.add_argument("--split_root", default=str(ROOT / "datasets/shapenet_split"))
    parser.add_argument(
        "--gs_root",
        default=os.environ.get("SHAPENET55GS_PLY_ROOT", "data/shapesplat_ply"),
    )
    return parser.parse_args()


def resolve_indices(dataset, indices, model_ids, count):
    if indices is not None:
        resolved = list(indices)
    elif model_ids is not None:
        stems = {Path(path).stem: index for index, path in enumerate(dataset.files)}
        resolved = []
        for model_id in model_ids:
            if model_id in stems:
                resolved.append(stems[model_id])
                continue
            matches = [index for stem, index in stems.items() if stem.endswith(model_id)]
            if len(matches) != 1:
                raise KeyError(
                    f"model ID {model_id!r} has {len(matches)} matches; "
                    "pass the full taxonomy-model ID"
                )
            resolved.append(matches[0])
    else:
        resolved = list(range(count))
    if not resolved:
        raise ValueError("at least one visualization object is required")
    if min(resolved) < 0 or max(resolved) >= len(dataset.files):
        raise IndexError(f"object indices out of range [0, {len(dataset.files) - 1}]")
    return resolved


def load_full_asset(gs_root, relative_path, device):
    vertex = IO.get(os.path.join(gs_root, relative_path))["vertex"]
    raw = read_gaussian_attribute(
        vertex, ["xyz", "opacity", "scale", "rotation", "sh"]
    )
    normalized, norm = normalize_eall(raw)
    gaussians = torch.from_numpy(normalized).float().to(device)
    stats = torch.tensor(
        [*norm.scale_center.tolist(), norm.scale_radius],
        dtype=torch.float32,
        device=device,
    )
    return gaussians, stats


def render_panel(gaussians, stats, device, center, radius, args, azimuth, color=None):
    fields = list(to_render_fields(gaussians, stats))
    if color is not None:
        fields[-1] = color
    return render_alpha(
        *fields,
        args.elevation,
        azimuth,
        args.distance,
        args.resolution,
        args.fov,
        [1.0, 1.0, 1.0],
        device,
        center=center,
        radius=radius,
        chunk_size=args.chunk_size,
        max_radius=args.max_splat_radius,
        alpha_threshold=args.alpha_threshold,
    )


def geometry_colors(prediction, partial):
    """Color observed/predicted regions without altering Gaussian geometry."""
    pairwise = torch.cdist(partial[:, :3], partial[:, :3])
    pairwise.fill_diagonal_(float("inf"))
    spacing = pairwise.amin(dim=1).median().clamp_min(1e-4)
    distance = torch.cdist(prediction[:, :3], partial[:, :3]).amin(dim=1)
    observed = distance <= 2.0 * spacing
    blue = torch.tensor([0.16, 0.48, 0.78], device=prediction.device)
    orange = torch.tensor([0.93, 0.45, 0.14], device=prediction.device)
    return torch.where(observed[:, None], blue, orange), float(2.0 * spacing)


def shared_foreground_crop(panel_sets, padding=0.04):
    """Crop every panel in a row with one union box, preserving fair scale."""
    height, width = panel_sets[0][0].shape[:2]
    union = np.zeros((height, width), dtype=bool)
    for panels in panel_sets:
        for image in panels:
            # The renderer uses a white background. Keep antialiased edge pixels.
            union |= np.any(image[..., :3] < 250, axis=-1)
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return panel_sets, [0, 0, width, height]

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * padding))
    x0, x1 = max(0, x0 - pad), min(width, x1 + pad)
    y0, y1 = max(0, y0 - pad), min(height, y1 + pad)
    cropped = [
        [image[y0:y1, x0:x1] for image in panels] for panels in panel_sets
    ]
    return cropped, [x0, y0, x1, y1]


def save_grid(rows, output, titles=PANEL_TITLES):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )
    row_ratios = [row[0].shape[0] / row[0].shape[1] for row in rows]
    column_width = 7.15 / 4.0
    figure_height = max(1.0, column_width * sum(row_ratios) + 0.18)
    figure = plt.figure(figsize=(7.15, figure_height))
    grid = figure.add_gridspec(
        len(rows), 4, height_ratios=row_ratios, wspace=0.006, hspace=0.012
    )
    axes = np.empty((len(rows), 4), dtype=object)
    for row_index, panels in enumerate(rows):
        for column, image in enumerate(panels):
            axes[row_index, column] = figure.add_subplot(grid[row_index, column])
            axes[row_index, column].imshow(image)
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column], fontsize=8.5, pad=2)
    figure.subplots_adjust(
        left=0.002,
        right=0.998,
        top=0.94,
        bottom=0.002,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.005)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.005)
    plt.close(figure)


def output_for_view(output, azimuth, multiple_views):
    if not multiple_views:
        return output
    label = f"{azimuth:g}".replace("-", "m").replace(".", "p")
    return output.with_name(f"{output.stem}_az{label}{output.suffix}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("paper-quality Full-GS rendering requires a CUDA GPU")
    device = torch.device("cuda")
    jepa, jepa_checkpoint = load_completion(args.jepa_checkpoint, device)
    mae, mae_checkpoint = load_completion(args.mae_checkpoint, device)
    jepa.eval()
    mae.eval()
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
        partial_points=sampling[0],
        target_points=sampling[1],
        visible_ratios=[args.visible_ratio],
        repeat_seeds=[args.case_seed],
        train=False,
    )
    indices = resolve_indices(dataset, args.indices, args.model_ids, args.num_examples)
    if args.object_azimuths is not None and len(args.object_azimuths) != len(indices):
        raise ValueError("--object_azimuths must contain one value per selected object")

    cached = []
    with torch.no_grad():
        for object_index in indices:
            sample = dataset[object_index]
            relative_path = dataset.files[object_index]
            partial = sample["partial"].to(device)
            target = sample["target"].to(device)
            prediction_mae = mae(partial.unsqueeze(0))[0]
            prediction_jepa = jepa(partial.unsqueeze(0))[0]
            full, full_stats = load_full_asset(args.gs_root, relative_path, device)
            center = 0.5 * (full[:, :3].amin(0) + full[:, :3].amax(0))
            radius = (full[:, :3] - center).norm(dim=-1).max()
            mae_geometry, geometry_threshold = geometry_colors(prediction_mae, partial)
            jepa_geometry, _ = geometry_colors(prediction_jepa, partial)
            cached.append(
                {
                    "object_index": object_index,
                    "relative_path": relative_path,
                    "taxonomy": sample["taxonomy"],
                    "model_id": sample["model_id"],
                    "partial": partial,
                    "target": target,
                    "mae": prediction_mae,
                    "jepa": prediction_jepa,
                    "full": full,
                    "stats": full_stats,
                    "center": center,
                    "radius": radius,
                    "mae_geometry": mae_geometry,
                    "jepa_geometry": jepa_geometry,
                    "geometry_threshold": geometry_threshold,
                }
            )

    output = Path(args.output).resolve()
    panel_root = (
        Path(args.panel_root).resolve()
        if args.panel_root
        else output.with_name(f"{output.stem}_panels")
    )
    manifest = {
        "version": 2,
        "renderer": "depth-sorted EWA alpha compositing",
        "jepa_checkpoint": str(Path(args.jepa_checkpoint).resolve()),
        "mae_checkpoint": str(Path(args.mae_checkpoint).resolve()),
        "jepa_method": jepa_checkpoint.get("method", "Gaussian-JEPA"),
        "mae_method": mae_checkpoint.get("method", "Gaussian-MAE"),
        "visible_ratio": args.visible_ratio,
        "case_seed": args.case_seed,
        "resolution": args.resolution,
        "elevation": args.elevation,
        "azimuths": args.azimuths,
        "object_azimuths": args.object_azimuths,
        "shared_crop": not args.no_shared_crop,
        "crop_padding": args.crop_padding,
        "panels_only": args.panels_only,
        "panel_root": str(panel_root),
        "objects": [],
        "figures": [],
    }
    for item in cached:
        manifest["objects"].append(
            {
                "object_index": item["object_index"],
                "relative_path": item["relative_path"],
                "taxonomy": item["taxonomy"],
                "model_id": item["model_id"],
                "source_gaussians": int(item["full"].shape[0]),
                "partial_gaussians": int(item["partial"].shape[0]),
                "target_gaussians": int(item["target"].shape[0]),
                "predicted_gaussians": int(item["jepa"].shape[0]),
                "geometry_threshold": item["geometry_threshold"],
            }
        )

    view_sets = [None] if args.object_azimuths is not None else args.azimuths
    for view in view_sets:
        rows = []
        geometry_rows = []
        view_metadata = []
        for row_index, item in enumerate(cached):
            azimuth = (
                args.object_azimuths[row_index]
                if args.object_azimuths is not None
                else float(view)
            )
            panels = [item["partial"], item["mae"], item["jepa"], item["full"]]
            rendered = [
                render_panel(
                    panel,
                    item["stats"],
                    device,
                    item["center"],
                    item["radius"],
                    args,
                    azimuth,
                )
                for panel in panels
            ]
            target_rendered = render_panel(
                item["target"],
                item["stats"],
                device,
                item["center"],
                item["radius"],
                args,
                azimuth,
            )
            appearance_export = [
                rendered[0], rendered[1], rendered[2], target_rendered, rendered[3]
            ]
            geometry = None
            geometry_export = None
            if not args.no_geometry_overlay:
                blue = torch.tensor([0.16, 0.48, 0.78], device=device)
                partial_color = blue.expand(item["partial"].shape[0], -1)
                full_color = torch.full(
                    (item["full"].shape[0], 3), 0.45, device=device
                )
                target_color = torch.full(
                    (item["target"].shape[0], 3), 0.45, device=device
                )
                geometry_export = [
                    render_panel(
                        item["partial"], item["stats"], device, item["center"],
                        item["radius"], args, azimuth, partial_color,
                    ),
                    render_panel(
                        item["mae"], item["stats"], device, item["center"],
                        item["radius"], args, azimuth, item["mae_geometry"],
                    ),
                    render_panel(
                        item["jepa"], item["stats"], device, item["center"],
                        item["radius"], args, azimuth, item["jepa_geometry"],
                    ),
                    render_panel(
                        item["target"], item["stats"], device, item["center"],
                        item["radius"], args, azimuth, target_color,
                    ),
                    render_panel(
                        item["full"], item["stats"], device, item["center"],
                        item["radius"], args, azimuth, full_color,
                    ),
                ]
                geometry = geometry_export[:3] + [geometry_export[4]]

            crop_box = [0, 0, args.resolution, args.resolution]
            if not args.no_shared_crop:
                panel_sets = [appearance_export]
                if geometry_export is not None:
                    panel_sets.append(geometry_export)
                panel_sets, crop_box = shared_foreground_crop(
                    panel_sets, args.crop_padding
                )
                appearance_export = panel_sets[0]
                rendered = appearance_export[:3] + [appearance_export[4]]
                if geometry_export is not None:
                    geometry_export = panel_sets[1]
                    geometry = geometry_export[:3] + [geometry_export[4]]

            rows.append(rendered)
            if geometry is not None:
                geometry_rows.append(geometry)
            angle_label = f"{azimuth:g}".replace("-", "m").replace(".", "p")
            category = TAXONOMY_NAMES.get(item["taxonomy"], item["taxonomy"])
            object_name = f"{category}__{item['model_id']}"
            object_dir = panel_root / object_name / f"azimuth_{angle_label}"
            appearance_dir = object_dir / "appearance"
            appearance_dir.mkdir(parents=True, exist_ok=True)
            for filename, image in zip(PANEL_FILES, appearance_export):
                Image.fromarray(image).save(appearance_dir / filename)
            if geometry_export is not None:
                geometry_dir = object_dir / "geometry"
                geometry_dir.mkdir(parents=True, exist_ok=True)
                for filename, image in zip(PANEL_FILES, geometry_export):
                    Image.fromarray(image).save(geometry_dir / filename)
            view_metadata.append(
                {
                    "model_id": item["model_id"],
                    "azimuth": azimuth,
                    "crop_box_xyxy": crop_box,
                }
            )

        if not args.panels_only:
            figure_output = output_for_view(output, view or 0.0, len(view_sets) > 1)
            save_grid(rows, figure_output)
            manifest["figures"].append(
                {"path": str(figure_output), "rows": view_metadata}
            )
            if geometry_rows:
                geometry_output = figure_output.with_name(
                    f"{figure_output.stem}_geometry{figure_output.suffix}"
                )
                save_grid(geometry_rows, geometry_output)
                manifest["figures"].append(
                    {"path": str(geometry_output), "rows": view_metadata}
                )

    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved {len(manifest['figures'])} figures, raw panels, and {manifest_path}")


if __name__ == "__main__":
    main()
