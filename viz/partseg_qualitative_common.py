"""Shared data and rendering utilities for PartSeg qualitative comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SEG_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}

# Paul Tol's bright palette, extended with a neutral orange.  Colors are
# indexed locally within each object category, so the same semantic label is
# identical for every method and the ground truth.
PART_PALETTE = (
    "#5B8DB8",  # denim blue
    "#5BBFA7",  # mint teal
    "#E9826B",  # soft coral
    "#E6BC55",  # warm gold
    "#9A80C5",  # lavender
    "#67B7D1",  # sky blue
)

DEFAULT_VIEWS = {
    "Airplane": ((32.0, 18.0), (145.0, 12.0)),
    "Chair": ((38.0, 18.0), (218.0, 15.0)),
    "Guitar": ((90.0, 10.0), (270.0, 10.0)),
    "Motorbike": ((35.0, 14.0), (215.0, 12.0)),
    "Table": ((38.0, 22.0), (220.0, 18.0)),
}


def stable_seed(object_key: str, seed: int) -> int:
    digest = hashlib.sha1(object_key.encode("utf8")).hexdigest()
    return (int(digest[:8], 16) + seed) % (2**32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    points -= points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    if radius > 0:
        points /= radius
    return points


def load_categories(data_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    category_to_synset: dict[str, str] = {}
    for line in (data_root / "synsetoffset2category.txt").read_text().splitlines():
        name, synset = line.split()
        category_to_synset[name] = synset
    category_to_index = {
        name: index for index, name in enumerate(category_to_synset)
    }
    return category_to_synset, category_to_index


def test_object_ids(data_root: Path) -> dict[str, list[str]]:
    split = json.loads(
        (
            data_root
            / "train_test_split"
            / "shuffled_test_file_list.json"
        ).read_text()
    )
    result: dict[str, list[str]] = {}
    for entry in split:
        parts = entry.replace("\\", "/").split("/")
        synset, object_id = parts[-2], parts[-1]
        result.setdefault(synset, []).append(Path(object_id).stem)
    return {key: sorted(set(value)) for key, value in result.items()}


def prepare_inputs(
    data_root: Path,
    output_root: Path,
    categories: list[str],
    per_category: int,
    num_points: int,
    seed: int,
) -> list[dict]:
    category_to_synset, category_to_index = load_categories(data_root)
    test_ids = test_object_ids(data_root)
    input_dir = output_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    for category in categories:
        synset = category_to_synset[category]
        ids = test_ids[synset]
        rng = np.random.RandomState(stable_seed(category, seed))
        chosen_ids = [ids[index] for index in rng.permutation(len(ids))[:per_category]]
        for object_id in chosen_ids:
            source = data_root / "shape_data" / synset / f"{object_id}.txt"
            data = np.loadtxt(source).astype(np.float32)
            points_full = data[:, :3]
            labels_full = data[:, -1].astype(np.int64)
            sample_rng = np.random.RandomState(
                stable_seed(f"{synset}-{object_id}", seed)
            )
            choice = sample_rng.choice(
                len(points_full), num_points, replace=True
            )
            points = points_full[choice]
            labels = labels_full[choice]
            key = f"{synset}-{object_id}"
            output = input_dir / f"{key}.npz"
            np.savez_compressed(
                output,
                points=points.astype(np.float32),
                points_normalized=normalize_points(points),
                labels=labels,
                choice=choice.astype(np.int64),
                category=np.asarray(category),
                category_index=np.asarray(category_to_index[category]),
                synset=np.asarray(synset),
                object_id=np.asarray(object_id),
                source=np.asarray(str(source)),
                sample_seed=np.asarray(stable_seed(key, seed)),
            )
            cases.append(
                {
                    "key": key,
                    "category": category,
                    "category_index": category_to_index[category],
                    "synset": synset,
                    "object_id": object_id,
                    "input": str(output),
                }
            )

    manifest = {
        "dataset": "ShapeNet-Part test split",
        "data_root": str(data_root),
        "num_points": num_points,
        "selection_seed": seed,
        "sampling": "fixed NumPy choice with replacement per object",
        "categories": categories,
        "per_category": per_category,
        "cases": cases,
    }
    (output_root / "cases_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return cases


def shape_iou(prediction: np.ndarray, labels: np.ndarray, category: str) -> float:
    values = []
    for part in SEG_CLASSES[category]:
        pred_mask = prediction == part
        label_mask = labels == part
        union = np.logical_or(pred_mask, label_mask).sum()
        if union == 0:
            values.append(1.0)
        else:
            values.append(
                np.logical_and(pred_mask, label_mask).sum() / float(union)
            )
    return float(np.mean(values))


def restrict_prediction(logits: np.ndarray, category: str) -> np.ndarray:
    parts = np.asarray(SEG_CLASSES[category], dtype=np.int64)
    return parts[np.argmax(logits[:, parts], axis=1)]


def rotation_matrix(azimuth: float, elevation: float) -> np.ndarray:
    az = math.radians(azimuth)
    el = math.radians(elevation)
    ry = np.asarray(
        [
            [math.cos(az), 0.0, math.sin(az)],
            [0.0, 1.0, 0.0],
            [-math.sin(az), 0.0, math.cos(az)],
        ],
        dtype=np.float32,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(el), -math.sin(el)],
            [0.0, math.sin(el), math.cos(el)],
        ],
        dtype=np.float32,
    )
    return rx @ ry


def _hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray(
        [int(value[index : index + 2], 16) for index in (0, 2, 4)],
        dtype=np.float32,
    )


def balanced_part_colors(
    category: str,
    counts: dict[int, int],
) -> dict[int, np.ndarray]:
    """Assign calm base colors to common parts and accents to small parts."""
    parts = SEG_CLASSES[category]
    ranked_parts = sorted(parts, key=lambda part: (-counts.get(part, 0), part))
    return {
        part: _hex_rgb(PART_PALETTE[index % len(PART_PALETTE)])
        for index, part in enumerate(ranked_parts)
    }


@lru_cache(maxsize=512)
def _sphere_sprite(
    radius: int,
    red: int,
    green: int,
    blue: int,
) -> Image.Image:
    """Create a softly lit sphere sprite for CPU point-cloud rendering."""
    radius = max(radius, 2)
    extent = radius + 2
    grid = np.arange(-extent, extent + 1, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid)
    nx = xx / float(radius)
    ny = yy / float(radius)
    squared_radius = nx**2 + ny**2
    inside = squared_radius <= 1.0
    nz = np.sqrt(np.clip(1.0 - squared_radius, 0.0, 1.0))

    light = np.asarray([-0.48, -0.58, 0.66], dtype=np.float32)
    light /= np.linalg.norm(light)
    diffuse = np.clip(
        nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0
    )
    rim = np.clip(nz, 0.0, 1.0)
    illumination = 0.64 + 0.29 * diffuse + 0.07 * rim

    # Restrained rough-plastic highlight, inspired by the reference Mitsuba
    # renderer without adding it as a runtime dependency.
    highlight = np.exp(
        -(((nx + 0.32) / 0.23) ** 2 + ((ny + 0.38) / 0.23) ** 2)
    )
    base = np.asarray([red, green, blue], dtype=np.float32)
    rgb = base[None, None, :] * illumination[..., None]
    rgb += 38.0 * highlight[..., None]
    rgb = np.clip(rgb, 0.0, 255.0)

    edge = np.clip((1.03 - np.sqrt(squared_radius)) * radius, 0.0, 1.0)
    alpha = np.where(inside, edge * 255.0, 0.0)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def render_surfels(
    points: np.ndarray,
    labels: np.ndarray,
    category: str,
    output: Path,
    *,
    azimuth: float,
    elevation: float,
    resolution: int = 1000,
    point_radius: float = 16.0,
    transparent: bool = True,
    part_colors: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    scale = 2
    size = resolution * scale
    points = normalize_points(points)
    camera = points @ rotation_matrix(azimuth, elevation).T
    xy = camera[:, :2]
    xy -= 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-6)
    image_scale = 0.82 * size / float(span.max())
    x = size / 2.0 + xy[:, 0] * image_scale
    y = size / 2.0 - xy[:, 1] * image_scale
    depth = camera[:, 2]
    depth_norm = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
    order = np.argsort(depth)

    background = (255, 255, 255, 0 if transparent else 255)
    canvas = Image.new("RGBA", (size, size), background)
    radius = point_radius * scale
    parts = SEG_CLASSES[category]
    part_to_color = (
        part_colors
        if part_colors is not None
        else {
            part: _hex_rgb(PART_PALETTE[index % len(PART_PALETTE)])
            for index, part in enumerate(parts)
        }
    )

    # Soft projected floor shadow supplies depth and a stable visual frame
    # without changing point locations or semantic colors.
    shadow_layer = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer, mode="RGBA")
    ground_y = min(size * 0.95, float(y.max()) + 0.025 * size)
    object_width = max(float(x.max() - x.min()), 1.0)
    shadow_draw.ellipse(
        (
            float(x.min()) - 0.04 * object_width,
            ground_y - 0.018 * size,
            float(x.max()) + 0.10 * object_width,
            ground_y + 0.030 * size,
        ),
        fill=(28, 38, 50, 21),
    )
    for index in order:
        height = max(0.0, ground_y - float(y[index]))
        shadow_x = float(x[index]) + 0.11 * height
        shadow_y = ground_y + 0.018 * height
        shadow_radius_x = radius * (0.78 + 0.08 * float(depth_norm[index]))
        shadow_radius_y = 0.30 * shadow_radius_x
        shadow_draw.ellipse(
            (
                shadow_x - shadow_radius_x,
                shadow_y - shadow_radius_y,
                shadow_x + shadow_radius_x,
                shadow_y + shadow_radius_y,
            ),
            fill=(28, 38, 50, 38),
        )
    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(radius=max(2.0, 0.55 * radius))
    )
    canvas.alpha_composite(shadow_layer)

    for index in order:
        base = part_to_color[int(labels[index])]
        base = 0.84 * base + 0.16 * 255.0
        depth_shade = 0.84 + 0.16 * float(depth_norm[index])
        color = np.clip(base * depth_shade, 0.0, 255.0).astype(np.uint8)
        point_r = int(
            round(radius * (0.94 + 0.08 * float(depth_norm[index])))
        )
        sprite = _sphere_sprite(point_r, *map(int, color))
        left = int(round(float(x[index]) - sprite.width / 2.0))
        top = int(round(float(y[index]) - sprite.height / 2.0))
        canvas.alpha_composite(sprite, (left, top))

    canvas = canvas.resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    if transparent:
        white = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
        white.alpha_composite(canvas)
        white.convert("RGB").save(
            output.with_name(output.stem + "_white" + output.suffix)
        )


def render_error(
    points: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    output: Path,
    *,
    azimuth: float,
    elevation: float,
) -> None:
    error_labels = np.where(prediction == labels, 0, 1)
    old = SEG_CLASSES.get("_Error")
    SEG_CLASSES["_Error"] = [0, 1]
    palette = PART_PALETTE
    try:
        globals()["PART_PALETTE"] = ("#C8CDD3", "#CC3311")
        render_surfels(
            points,
            error_labels,
            "_Error",
            output,
            azimuth=azimuth,
            elevation=elevation,
        )
    finally:
        globals()["PART_PALETTE"] = palette
        if old is None:
            SEG_CLASSES.pop("_Error", None)
        else:
            SEG_CLASSES["_Error"] = old
