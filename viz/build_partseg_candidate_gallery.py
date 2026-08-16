#!/usr/bin/env python3
"""Build compact browsing sheets from rendered PartSeg candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Union

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/partseg_qualitative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--top-page-size", type=int, default=5)
    return parser.parse_args()


def font(size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.is_file()
        else ImageFont.load_default()
    )


def load_preview(root: Path, key: str) -> Image.Image:
    matches = sorted((root / "previews").glob(f"{key}_view1_*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one primary preview for {key}: {matches}")
    image = Image.open(matches[0]).convert("RGB")
    image.thumbnail((1800, 520), Image.Resampling.LANCZOS)
    return image


def case_strip(root: Path, case: dict, width: int) -> Image.Image:
    preview = load_preview(root, case["key"])
    preview.thumbnail((width, 520), Image.Resampling.LANCZOS)
    header_height = 54
    strip = Image.new("RGB", (width, header_height + preview.height), "white")
    draw = ImageDraw.Draw(strip)
    gain = 100.0 * case["ours_minus_gaussian_mae"]
    title = (
        f"{case['category']}  {case['object_id']}    "
        f"Gaussian-JEPA - Gaussian-MAE: {gain:+.1f} IoU"
    )
    draw.text((18, 12), title, fill=(30, 30, 30), font=font(25))
    strip.paste(preview, ((width - preview.width) // 2, header_height))
    return strip


def stack_cases(root: Path, cases: List[dict], width: int, output: Path) -> None:
    strips = [case_strip(root, case, width) for case in cases]
    gap = 12
    canvas = Image.new(
        "RGB",
        (width, sum(strip.height for strip in strips) + gap * (len(strips) - 1)),
        "white",
    )
    y = 0
    for strip in strips:
        canvas.paste(strip, (0, y))
        y += strip.height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)


def main() -> None:
    args = parse_args()
    summary_path = args.input_root / "qualitative_summary.json"
    summary = json.loads(summary_path.read_text())
    cases = summary["cases"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case["category"]].append(case)

    gallery_root = args.input_root / "galleries"
    for category, category_cases in sorted(grouped.items()):
        ranked = sorted(
            category_cases,
            key=lambda case: case["ours_minus_gaussian_mae"],
            reverse=True,
        )
        stack_cases(
            args.input_root,
            ranked,
            args.width,
            gallery_root / f"{category.lower()}_all_cases.jpg",
        )

    ranked_all = sorted(
        cases,
        key=lambda case: case["ours_minus_gaussian_mae"],
        reverse=True,
    )
    for start in range(0, len(ranked_all), args.top_page_size):
        page = start // args.top_page_size + 1
        stack_cases(
            args.input_root,
            ranked_all[start : start + args.top_page_size],
            args.width,
            gallery_root / f"ranked_page_{page:02d}.jpg",
        )

    print(
        f"Created {len(grouped)} category sheets and "
        f"{(len(ranked_all) + args.top_page_size - 1) // args.top_page_size} "
        f"ranked pages in {gallery_root}"
    )


if __name__ == "__main__":
    main()
