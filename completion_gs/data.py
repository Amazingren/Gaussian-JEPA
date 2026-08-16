"""Deterministic partial/full ShapeNet55-GS pairs without target-position leakage."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.ShapeNet55Gaussian import read_gaussian_attribute
from datasets.io import IO


SH_C0 = 0.28209479177387814


def _seed(*values: object) -> int:
    text = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class GaussianNorm:
    """Object-level statistics for the exact E(All) preprocessing."""

    scale_center: np.ndarray
    scale_radius: float


def normalize_eall(raw: np.ndarray) -> tuple[np.ndarray, GaussianNorm]:
    """Match ShapeNet55-GS E(All) normalization, computed before sampling."""
    data = raw.astype(np.float32, copy=True)
    xyz = data[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    radius = float(np.linalg.norm(xyz, axis=1).max())
    radius = max(radius, 1e-8)
    data[:, :3] = xyz / radius
    data[:, 4:7] /= radius

    data[:, 3] = data[:, 3] * 2.0 - 1.0
    scale_center = data[:, 4:7].mean(axis=0).astype(np.float32)
    scale_centered = data[:, 4:7] - scale_center
    scale_radius = float(np.linalg.norm(scale_centered, axis=1).max())
    scale_radius = max(scale_radius, 1e-8)
    data[:, 4:7] = scale_centered / scale_radius

    data[:, 11:14] = np.clip(data[:, 11:14] * SH_C0, -0.5, 0.5)
    data[:, 11:14] *= 2.0 / np.sqrt(3.0)
    return data, GaussianNorm(scale_center=scale_center, scale_radius=scale_radius)


def to_render_fields(normalized: torch.Tensor, stats: torch.Tensor):
    """Convert normalized E(All) tensors to unit-frame renderer fields.

    stats is (..., 4): normalized scale center xyz followed by scale radius.
    """
    while stats.ndim < normalized.ndim:
        stats = stats.unsqueeze(-2)
    xyz = normalized[..., :3]
    opacity = ((normalized[..., 3] + 1.0) * 0.5).clamp(0.0, 1.0)
    scale = normalized[..., 4:7] * stats[..., 3:4] + stats[..., :3]
    scale = scale.clamp_min(1e-4)
    quat = normalized[..., 7:11]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    color = (0.5 + normalized[..., 11:14] * (np.sqrt(3.0) * 0.5)).clamp(0.0, 1.0)
    return xyz, scale, quat, opacity, color


class ShapeNetCompletionDataset(Dataset):
    """Return a spatially partial input and an independently sampled full target.

    The crop is selected from the complete asset by a random half-space direction.
    Only the retained primitives are returned to the encoder. The target sampling is
    independent and no missing-region centers are returned.
    """

    attributes = ["xyz", "opacity", "scale", "rotation", "sh"]

    def __init__(
        self,
        split_file: str,
        gs_root: str,
        partial_points: int = 512,
        target_points: int = 1024,
        visible_ratios: Iterable[float] = (0.3, 0.5, 0.7),
        seed: int = 0,
        train: bool = False,
        repeat_seeds: Iterable[int] = (0,),
        indices: Optional[Iterable[int]] = None,
    ):
        with open(split_file, "r", encoding="utf-8") as handle:
            files = [line.strip() for line in handle if line.strip()]
        if indices is not None:
            files = [files[index] for index in indices]
        self.files = files
        self.gs_root = gs_root
        self.partial_points = int(partial_points)
        self.target_points = int(target_points)
        self.visible_ratios = tuple(float(value) for value in visible_ratios)
        self.seed = int(seed)
        self.train = bool(train)
        self.repeat_seeds = tuple(int(value) for value in repeat_seeds)
        self.epoch = 0
        if not self.visible_ratios or any(not 0.0 < value <= 1.0 for value in self.visible_ratios):
            raise ValueError(f"invalid visible ratios: {self.visible_ratios}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.train:
            return len(self.files)
        return len(self.files) * len(self.visible_ratios) * len(self.repeat_seeds)

    def _case(self, index: int) -> tuple[int, float, int]:
        if self.train:
            ratio_index = _seed(self.seed, self.epoch, index, "ratio") % len(self.visible_ratios)
            return index, self.visible_ratios[ratio_index], self.epoch
        per_object = len(self.visible_ratios) * len(self.repeat_seeds)
        object_index, case_index = divmod(index, per_object)
        ratio_index, seed_index = divmod(case_index, len(self.repeat_seeds))
        return object_index, self.visible_ratios[ratio_index], self.repeat_seeds[seed_index]

    def __getitem__(self, index: int):
        object_index, visible_ratio, case_seed = self._case(index)
        rel_path = self.files[object_index]
        vertex = IO.get(os.path.join(self.gs_root, rel_path))["vertex"]
        raw = read_gaussian_attribute(vertex, self.attributes)
        normalized, norm = normalize_eall(raw)

        rng = np.random.default_rng(_seed(self.seed, case_seed, object_index, visible_ratio))
        direction = rng.normal(size=3).astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        scores = normalized[:, :3] @ direction
        visible_count = max(1, int(round(len(normalized) * visible_ratio)))
        visible_count = min(visible_count, len(normalized))
        visible_indices = np.argpartition(scores, -visible_count)[-visible_count:]

        partial_indices = rng.choice(
            visible_indices, self.partial_points, replace=visible_count < self.partial_points
        )
        target_indices = rng.choice(
            len(normalized), self.target_points, replace=len(normalized) < self.target_points
        )
        partial = torch.from_numpy(normalized[partial_indices].copy()).float()
        target = torch.from_numpy(normalized[target_indices].copy()).float()
        stats = torch.tensor(
            [*norm.scale_center.tolist(), norm.scale_radius], dtype=torch.float32
        )
        stem = os.path.splitext(os.path.basename(rel_path))[0]
        taxonomy = rel_path.split("-")[0]
        return {
            "partial": partial,
            "target": target,
            "stats": stats,
            "taxonomy": taxonomy,
            "model_id": stem,
            "visible_ratio": torch.tensor(visible_ratio, dtype=torch.float32),
            "case_seed": torch.tensor(case_seed, dtype=torch.long),
        }


def split_train_indices(count: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    val_count = max(1, int(round(count * val_fraction)))
    return order[val_count:].tolist(), order[:val_count].tolist()
