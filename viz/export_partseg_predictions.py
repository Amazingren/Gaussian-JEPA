#!/usr/bin/env python3
"""Export deterministic PartSeg predictions for one supported method.

Each invocation runs in a fresh process to isolate the three upstream code
bases and their modules.  The output point indices come from the shared case
manifest, so every method is visualized on exactly the same labeled points.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
POINT_MAE_ROOT = Path(os.environ.get("POINT_MAE_ROOT", ROOT.parent / "Point-MAE"))
POINT_JEPA_ROOT = Path(os.environ.get("POINT_JEPA_ROOT", ROOT.parent / "Point-JEPA"))
GAUSSIAN_MAE_ROOT = Path(
    os.environ.get("GAUSSIAN_MAE_ROOT", ROOT.parent / "ShapeSplat-Gaussian_MAE")
)
DEFAULT_OUTPUT = ROOT / "outputs/partseg_qualitative"
DEFAULT_DATA = Path(os.environ.get("PARTANNO_ROOT", "data/shapenet_part"))
DEFAULT_GS = Path(os.environ.get("SHAPENET55GS_PLY_ROOT", "data/shapesplat_ply"))

sys.path.insert(0, str(ROOT / "viz"))
from partseg_qualitative_common import (  # noqa: E402
    SEG_CLASSES,
    restrict_prediction,
    shape_iou,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("point_mae", "point_jepa", "gaussian_mae", "gaussian_jepa"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS)
    parser.add_argument(
        "--pc-to-gs-map",
        type=Path,
        default=ROOT / "segmentation_gs/split_to_org_gs_map.json",
    )
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pure_fps_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    batch, num_points, _ = points.shape
    selected = torch.empty(batch, count, dtype=torch.long, device=points.device)
    distance = torch.full(
        (batch, num_points),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )
    centroid = points.mean(dim=1, keepdim=True)
    farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    batch_indices = torch.arange(batch, device=points.device)
    for index in range(count):
        selected[:, index] = farthest
        current = points[batch_indices, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((points - current) ** 2).sum(dim=-1))
        farthest = distance.argmax(dim=1)
    return selected


def gather_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        features,
        2,
        indices.unsqueeze(1).expand(-1, features.shape[1], -1),
    )


class PureTorchKNN:
    def __init__(self, k: int, transpose_mode: bool = True):
        self.k = k
        self.transpose_mode = transpose_mode

    def __call__(
        self,
        reference: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.transpose_mode:
            reference = reference.transpose(1, 2)
            query = query.transpose(1, 2)
        distances = torch.cdist(query.float(), reference.float())
        values, indices = torch.topk(
            distances,
            self.k,
            dim=-1,
            largest=False,
            sorted=False,
        )
        return values, indices


def install_cuda_extension_stubs() -> None:
    """Let upstream models import while executing grouping in pure PyTorch."""

    pointnet_utils = types.ModuleType("pointnet2_utils_stub")
    pointnet_utils.furthest_point_sample = pure_fps_indices
    pointnet_utils.gather_operation = gather_operation

    pointnet_package = types.ModuleType("pointnet2_ops")
    pointnet_package.pointnet2_utils = pointnet_utils
    sys.modules["pointnet2_ops"] = pointnet_package
    sys.modules["pointnet2_ops.pointnet2_utils"] = pointnet_utils

    knn_module = types.ModuleType("knn_cuda")
    knn_module.KNN = PureTorchKNN
    sys.modules["knn_cuda"] = knn_module


def case_manifest(output_root: Path) -> list[dict]:
    path = output_root / "cases_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run export_partseg_qualitative.py prepare first"
        )
    return json.loads(path.read_text())["cases"]


def checkpoint_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "base_model" in checkpoint:
        return checkpoint["base_model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_point_mae(path: Path) -> torch.nn.Module:
    install_cuda_extension_stubs()
    sys.path.insert(0, str(POINT_MAE_ROOT / "segmentation"))
    model_dir = POINT_MAE_ROOT / "segmentation/models"
    sys.path.insert(0, str(model_dir))
    import pt  # type: ignore

    model = pt.get_model(50)
    incompatible = model.load_state_dict(checkpoint_state(path), strict=False)
    allowed_unexpected = {
        key for key in incompatible.unexpected_keys if key.startswith("conv_fuse.")
    }
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed_unexpected:
        raise RuntimeError(str(incompatible))
    return model.eval()


class _PointJepaDataModule:
    num_classes = 16
    num_seg_classes = 50
    category_to_seg_classes = SEG_CLASSES
    seg_class_to_category = {
        part: category
        for category, parts in SEG_CLASSES.items()
        for part in parts
    }


def load_point_jepa(path: Path) -> torch.nn.Module:
    import importlib.util

    sys.path.insert(0, str(POINT_JEPA_ROOT))
    scheduler_module = types.ModuleType("pl_bolts.optimizers.lr_scheduler")
    scheduler_module.LinearWarmupCosineAnnealingLR = object
    optimizer_module = types.ModuleType("pl_bolts.optimizers")
    optimizer_module.lr_scheduler = scheduler_module
    bolts_module = types.ModuleType("pl_bolts")
    bolts_module.optimizers = optimizer_module
    sys.modules.setdefault("pl_bolts", bolts_module)
    sys.modules.setdefault("pl_bolts.optimizers", optimizer_module)
    sys.modules.setdefault(
        "pl_bolts.optimizers.lr_scheduler", scheduler_module
    )
    import torchmetrics

    class InferenceAccuracy(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    torchmetrics.Accuracy = InferenceAccuracy
    source = POINT_JEPA_ROOT / "pointjepa/models/part_segmentation.py"
    module = types.ModuleType("_pointjepa_part_segmentation")
    module.__file__ = str(source)
    code = "from __future__ import annotations\n" + source.read_text()
    exec(compile(code, str(source), "exec"), module.__dict__)
    PointJepaPartSegmentation = module.PointJepaPartSegmentation

    checkpoint = torch.load(path, map_location="cpu")
    hparams = dict(checkpoint["hyper_parameters"])
    hparams["pretrained_ckpt_path"] = None
    model = PointJepaPartSegmentation(**hparams)
    model._trainer = SimpleNamespace(
        datamodule=_PointJepaDataModule(),
        loggers=[],
        current_epoch=0,
        max_epochs=300,
    )
    model.setup()
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    return model.eval()


def gaussian_args() -> SimpleNamespace:
    return SimpleNamespace(
        num_group=128,
        group_attribute=["xyz"],
        attribute=["xyz", "opacity", "scale", "rotation", "sh"],
        soft_knn=False,
    )


def load_gaussian_model(method: str, path: Path) -> torch.nn.Module:
    install_cuda_extension_stubs()
    repo = ROOT if method == "gaussian_jepa" else GAUSSIAN_MAE_ROOT
    model_dir = repo / "segmentation_gs/models"
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(repo / "segmentation_gs"))
    import pt  # type: ignore

    model = pt.get_model(50, gaussian_args())
    incompatible = model.load_state_dict(checkpoint_state(path), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    return model.eval()


def one_hot(category_index: int) -> torch.Tensor:
    return torch.eye(16, dtype=torch.float32)[category_index].unsqueeze(0)


def read_gaussians(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float32)))
    scale_names = sorted(
        [name for name in vertex.dtype.names if name.startswith("scale_")],
        key=lambda value: int(value.split("_")[-1]),
    )
    scales = np.exp(
        np.stack([vertex[name].astype(np.float32) for name in scale_names], axis=1)
    )
    rotation_names = sorted(
        [name for name in vertex.dtype.names if name.startswith("rot")],
        key=lambda value: int(value.split("_")[-1]),
    )
    rotations = np.stack(
        [vertex[name].astype(np.float32) for name in rotation_names], axis=1
    )
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-9
    rotations *= np.sign(rotations[:, :1])
    sh = np.stack(
        [vertex[f"f_dc_{index}"].astype(np.float32) for index in range(3)],
        axis=1,
    )
    return np.concatenate(
        [
            np.stack([x, y, z], axis=1),
            opacity[:, None],
            scales,
            rotations,
            sh,
        ],
        axis=1,
    ).astype(np.float32)


def normalize_gaussians(
    gaussians: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gaussians = gaussians.copy()
    points = points.copy()
    centroid = gaussians[:, :3].mean(axis=0)
    gaussians[:, :3] -= centroid
    radius = np.linalg.norm(gaussians[:, :3], axis=1).max()
    gaussians[:, :3] /= radius
    gaussians[:, 4:7] /= radius
    points = (points - centroid) / radius

    gaussians[:, 3] = gaussians[:, 3] * 2.0 - 1.0
    scale_center = gaussians[:, 4:7].mean(axis=0)
    gaussians[:, 4:7] -= scale_center
    scale_radius = np.linalg.norm(gaussians[:, 4:7], axis=1).max()
    if scale_radius > 0:
        gaussians[:, 4:7] /= scale_radius
    sh = gaussians[:, 11:14] * 0.28209479177387814
    sh = np.clip(sh, -0.5, 0.5)
    gaussians[:, 11:14] = 2.0 * sh / math.sqrt(3.0)
    return gaussians, points


def gaussian_input(
    case: dict,
    input_data: np.lib.npyio.NpzFile,
    gs_root: Path,
    mapping: dict[str, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    key = case["key"]
    if key not in mapping:
        raise KeyError(f"No Gaussian mapping for {key}")
    gs_path = gs_root / mapping[key]
    gaussians = read_gaussians(gs_path)
    points = input_data["points"].astype(np.float32)
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    points = points @ rotation
    gaussians, points = normalize_gaussians(gaussians, points)
    rng = np.random.RandomState(stable_seed(key + "-gs", 2027))
    choice = rng.choice(len(gaussians), 2048, replace=True)
    gs_tensor = torch.from_numpy(gaussians[choice]).unsqueeze(0).transpose(1, 2)
    point_tensor = torch.from_numpy(points).unsqueeze(0)
    return gs_tensor.float(), point_tensor.float()


def predict_case(
    method: str,
    model: torch.nn.Module,
    case: dict,
    data: np.lib.npyio.NpzFile,
    gs_root: Path,
    mapping: Optional[Dict[str, str]],
) -> np.ndarray:
    category = str(data["category"])
    category_index = int(data["category_index"])
    points = torch.from_numpy(data["points_normalized"]).unsqueeze(0).float()
    torch.manual_seed(stable_seed(case["key"] + method, 2027))
    with torch.inference_mode():
        if method == "point_mae":
            logits = model(points.transpose(1, 2), one_hot(category_index))
        elif method == "point_jepa":
            logits = model(points, torch.tensor([category_index]))
        else:
            assert mapping is not None
            gs_data, pc_xyz = gaussian_input(case, data, gs_root, mapping)
            logits = model(gs_data, one_hot(category_index), pc_xyz)
    return restrict_prediction(logits[0].cpu().numpy(), category)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    if args.method == "point_mae":
        model = load_point_mae(args.checkpoint)
    elif args.method == "point_jepa":
        model = load_point_jepa(args.checkpoint)
    else:
        model = load_gaussian_model(args.method, args.checkpoint)

    mapping = None
    if args.method.startswith("gaussian"):
        mapping = json.loads(args.pc_to_gs_map.read_text())

    output_dir = args.output_root / "predictions" / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, case in enumerate(case_manifest(args.output_root), start=1):
        output = output_dir / f"{case['key']}.npz"
        if output.exists() and not args.overwrite:
            with np.load(output) as cached:
                summary.append(
                    {
                        "key": case["key"],
                        "category": case["category"],
                        "iou": float(cached["iou"]),
                    }
                )
            continue
        with np.load(case["input"]) as data:
            prediction = predict_case(
                args.method,
                model,
                case,
                data,
                args.gs_root,
                mapping,
            )
            labels = data["labels"]
            iou = shape_iou(prediction, labels, case["category"])
            np.savez_compressed(
                output,
                prediction=prediction.astype(np.int64),
                labels=labels.astype(np.int64),
                iou=np.asarray(iou, dtype=np.float32),
                method=np.asarray(args.method),
                key=np.asarray(case["key"]),
            )
        summary.append(
            {"key": case["key"], "category": case["category"], "iou": iou}
        )
        print(
            f"[{index}/{len(case_manifest(args.output_root))}] "
            f"{args.method}: {case['key']} IoU={iou:.4f}",
            flush=True,
        )

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "checkpoint": str(args.checkpoint),
                "cases": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
