#!/usr/bin/env python3
"""Prepare and render paper-quality ShapeNet-Part qualitative comparisons."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "viz"))
from partseg_qualitative_common import (  # noqa: E402
    DEFAULT_VIEWS,
    SEG_CLASSES,
    balanced_part_colors,
    prepare_inputs,
    render_error,
    render_surfels,
)


DEFAULT_DATA = Path(os.environ.get("PARTANNO_ROOT", "data/shapenet_part"))
DEFAULT_OUTPUT = ROOT / "outputs/partseg_qualitative"
METHODS = ("point_mae", "point_jepa", "gaussian_mae", "gaussian_jepa")
DISPLAY_NAMES = {
    "point_mae": "Point-MAE",
    "point_jepa": "Point-JEPA",
    "gaussian_mae": "Gaussian-MAE",
    "gaussian_jepa": "Gaussian-JEPA (Ours)",
    "ground_truth": "Ground Truth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument(
        "--categories",
        nargs="+",
        default=("Airplane", "Chair", "Guitar", "Motorbike", "Table"),
    )
    prepare.add_argument("--per-category", type=int, default=4)
    prepare.add_argument("--num-points", type=int, default=2048)
    prepare.add_argument("--seed", type=int, default=2027)

    render = subparsers.add_parser("render")
    render.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    render.add_argument("--resolution", type=int, default=1000)
    render.add_argument("--point-radius", type=float, default=16.0)
    render.add_argument("--allow-missing", action="store_true")
    render.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(root: Path) -> dict:
    path = root / "cases_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_prediction(
    root: Path, method: str, key: str
) -> Optional[Tuple[np.ndarray, float]]:
    path = root / "predictions" / method / f"{key}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return data["prediction"].copy(), float(data["iou"])


def labeled_preview(
    panels: List[Tuple[str, Image.Image, Optional[float]]],
    output: Path,
) -> None:
    tile = 500
    label_height = 72
    margin = 12
    canvas = Image.new(
        "RGB",
        (len(panels) * tile + (len(panels) + 1) * margin, tile + label_height + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    font = (
        ImageFont.truetype(str(font_path), 21)
        if font_path.is_file()
        else ImageFont.load_default()
    )
    for index, (name, image, iou) in enumerate(panels):
        x = margin + index * (tile + margin)
        image = image.convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(
            image,
            (x + (tile - image.width) // 2, label_height + margin),
        )
        label = DISPLAY_NAMES[name]
        if iou is not None:
            label += f"  IoU {iou * 100:.1f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x + (tile - text_width) / 2.0, 18),
            label,
            fill=(25, 25, 25),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_all(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.output_root)
    category_counts = {
        category: {part: 0 for part in parts}
        for category, parts in SEG_CLASSES.items()
    }
    for case in manifest["cases"]:
        with np.load(case["input"]) as data:
            labels = data["labels"]
        for part in SEG_CLASSES[case["category"]]:
            category_counts[case["category"]][part] += int(
                np.count_nonzero(labels == part)
            )
    category_colors = {
        category: balanced_part_colors(category, counts)
        for category, counts in category_counts.items()
    }

    summary = []
    for case_index, case in enumerate(manifest["cases"], start=1):
        with np.load(case["input"]) as data:
            points = data["points_normalized"]
            labels = data["labels"]
        predictions = {
            method: load_prediction(args.output_root, method, case["key"])
            for method in METHODS
        }
        missing = [method for method, result in predictions.items() if result is None]
        if missing and not args.allow_missing:
            raise FileNotFoundError(
                f"Missing predictions for {case['key']}: {', '.join(missing)}"
            )

        category = case["category"]
        views = DEFAULT_VIEWS.get(category, ((35.0, 18.0), (215.0, 15.0)))
        case_summary = {
            "key": case["key"],
            "category": category,
            "object_id": case["object_id"],
            "ious": {
                method: None if result is None else result[1]
                for method, result in predictions.items()
            },
        }
        summary.append(case_summary)

        for view_index, (azimuth, elevation) in enumerate(views, start=1):
            view_name = f"view{view_index}_az{int(azimuth):03d}_el{int(elevation):02d}"
            output_dir = (
                args.output_root
                / "assets"
                / category.lower()
                / case["key"]
                / view_name
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            panels: List[Tuple[str, Image.Image, Optional[float]]] = []
            for method in METHODS:
                result = predictions[method]
                if result is None:
                    continue
                prediction, iou = result
                semantic = output_dir / f"{method}.png"
                error = output_dir / f"{method}_error.png"
                if args.overwrite or not semantic.is_file():
                    render_surfels(
                        points,
                        prediction,
                        category,
                        semantic,
                        azimuth=azimuth,
                        elevation=elevation,
                        resolution=args.resolution,
                        point_radius=args.point_radius,
                        part_colors=category_colors[category],
                    )
                    render_error(
                        points,
                        prediction,
                        labels,
                        error,
                        azimuth=azimuth,
                        elevation=elevation,
                    )
                panels.append(
                    (method, Image.open(semantic.with_name(semantic.stem + "_white.png")), iou)
                )

            gt = output_dir / "ground_truth.png"
            if args.overwrite or not gt.is_file():
                render_surfels(
                    points,
                    labels,
                    category,
                    gt,
                    azimuth=azimuth,
                    elevation=elevation,
                    resolution=args.resolution,
                    point_radius=args.point_radius,
                    part_colors=category_colors[category],
                )
            panels.append(
                (
                    "ground_truth",
                    Image.open(gt.with_name(gt.stem + "_white.png")),
                    None,
                )
            )
            labeled_preview(
                panels,
                args.output_root / "previews" / f"{case['key']}_{view_name}.png",
            )
        print(
            f"[{case_index}/{len(manifest['cases'])}] Rendered {case['key']}",
            flush=True,
        )

    # Sort by the Gaussian-JEPA gain over Gaussian-MAE once both are present.
    for item in summary:
        ours = item["ious"]["gaussian_jepa"]
        baseline = item["ious"]["gaussian_mae"]
        item["ours_minus_gaussian_mae"] = (
            None if ours is None or baseline is None else ours - baseline
        )
    ranked = sorted(
        summary,
        key=lambda item: (
            item["ours_minus_gaussian_mae"] is not None,
            item["ours_minus_gaussian_mae"]
            if item["ours_minus_gaussian_mae"] is not None
            else -1e9,
        ),
        reverse=True,
    )
    (args.output_root / "qualitative_summary.json").write_text(
        json.dumps({"cases": summary, "ranked": ranked}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        cases = prepare_inputs(
            args.data_root,
            args.output_root,
            list(args.categories),
            args.per_category,
            args.num_points,
            args.seed,
        )
        print(f"Prepared {len(cases)} deterministic cases in {args.output_root}")
    else:
        render_all(args)


if __name__ == "__main__":
    main()
