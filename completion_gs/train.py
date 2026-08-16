"""Train/evaluate a shared completion decoder on a frozen JEPA or MAE backbone."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from completion_gs.data import ShapeNetCompletionDataset, split_train_indices
from completion_gs.model import GaussianCompletion, chamfer_and_matches, completion_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_prefix", required=True, choices=["JEPA_encoder.", "teacher_encoder.", "MAE_encoder."])
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_root", default=str(ROOT / "datasets/shapenet_split"))
    parser.add_argument(
        "--gs_root",
        default=os.environ.get("SHAPENET55GS_PLY_ROOT", "data/shapesplat_ply"),
    )
    parser.add_argument("--partial_points", type=int, default=512)
    parser.add_argument("--target_points", type=int, default=1024)
    parser.add_argument("--train_visible_ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--test_visible_ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--test_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--attr_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_train_objects", type=int, default=0)
    parser.add_argument("--max_test_objects", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def make_datasets(args):
    train_file = os.path.join(args.split_root, "train.txt")
    test_file = os.path.join(args.split_root, "test.txt")
    train_indices, val_indices = split_train_indices(
        count_lines(train_file), args.val_fraction, args.seed
    )
    if args.max_train_objects:
        train_indices = train_indices[: args.max_train_objects]
        val_indices = val_indices[: max(16, min(len(val_indices), args.max_train_objects // 4))]
    test_indices = None
    if args.max_test_objects:
        test_indices = list(range(min(count_lines(test_file), args.max_test_objects)))
    common = dict(
        gs_root=args.gs_root,
        partial_points=args.partial_points,
        target_points=args.target_points,
        seed=args.seed,
    )
    train = ShapeNetCompletionDataset(
        train_file,
        visible_ratios=args.train_visible_ratios,
        train=True,
        indices=train_indices,
        **common,
    )
    validation = ShapeNetCompletionDataset(
        train_file,
        visible_ratios=[0.5],
        repeat_seeds=[0],
        indices=val_indices,
        **common,
    )
    test = ShapeNetCompletionDataset(
        test_file,
        visible_ratios=args.test_visible_ratios,
        repeat_seeds=args.test_seeds,
        indices=test_indices,
        **common,
    )
    return train, validation, test


def make_loader(dataset, batch_size, workers, shuffle=False, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=True,
        # Workers are recreated each epoch so Dataset.set_epoch() reaches them.
        persistent_workers=False,
    )


def learning_rate_scale(epoch: int, warmup: int, total: int):
    if epoch < warmup:
        return float(epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, epoch, best, args):
    torch.save(
        {
            "version": 2,
            "epoch": epoch,
            "best_val_chamfer": best,
            "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "encoder_checkpoint": os.path.abspath(args.encoder_checkpoint),
            "encoder_prefix": args.encoder_prefix,
            "method": args.method,
            "args": vars(args),
        },
        path,
    )


def load_completion(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GaussianCompletion(
        checkpoint["encoder_checkpoint"], checkpoint["encoder_prefix"]
    ).to(device)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    return model, checkpoint


def train_epoch(model, loader, optimizer, device, attr_weight, epoch):
    model.train()
    sums = defaultdict(float)
    count = 0
    start = time.time()
    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(partial)
        loss, metrics = completion_loss(prediction, target, attr_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 10.0)
        optimizer.step()
        batch_size = partial.shape[0]
        count += batch_size
        for key, value in metrics.items():
            sums[key] += float(value) * batch_size
    result = {key: value / count for key, value in sums.items()}
    result["seconds"] = time.time() - start
    result["epoch"] = epoch
    return result


@torch.no_grad()
def evaluate(model, loader, device, output_csv: Optional[str] = None):
    model.eval()
    totals = defaultdict(float)
    ratio_totals = defaultdict(lambda: defaultdict(float))
    records = []
    for batch in loader:
        partial = batch["partial"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = model(partial)
        _, pred_dist, target_dist, _ = chamfer_and_matches(
            prediction[..., :3], target[..., :3]
        )
        chamfer = pred_dist.mean(1) + target_dist.mean(1)
        precision = (pred_dist < 0.01).float().mean(1)
        recall = (target_dist < 0.01).float().mean(1)
        fscore = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        for index in range(partial.shape[0]):
            ratio = float(batch["visible_ratio"][index])
            values = {
                "chamfer": float(chamfer[index]),
                "fscore": float(fscore[index]),
            }
            totals["count"] += 1
            ratio_totals[ratio]["count"] += 1
            for key, value in values.items():
                totals[key] += value
                ratio_totals[ratio][key] += value
            if output_csv is not None:
                records.append(
                    {
                        "taxonomy": batch["taxonomy"][index],
                        "model_id": batch["model_id"][index],
                        "visible_ratio": ratio,
                        "case_seed": int(batch["case_seed"][index]),
                        **values,
                    }
                )
    overall = {
        key: value / totals["count"] for key, value in totals.items() if key != "count"
    }
    overall["count"] = int(totals["count"])
    by_ratio = {}
    for ratio, values in sorted(ratio_totals.items()):
        by_ratio[f"{ratio:.2f}"] = {
            key: value / values["count"] for key, value in values.items() if key != "count"
        }
        by_ratio[f"{ratio:.2f}"]["count"] = int(values["count"])
    if output_csv is not None:
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {"overall": overall, "by_visible_ratio": by_ratio}


def main():
    args = parse_args()
    if args.target_points != 1024:
        raise ValueError("the shared decoder predicts exactly 1024 target Gaussians")
    if args.smoke:
        args.epochs = 1
        args.max_train_objects = 64
        args.max_test_objects = 8
        args.batch_size = min(args.batch_size, 8)
        args.eval_batch_size = min(args.eval_batch_size, 4)
        args.num_workers = 0
    set_seed(args.seed)
    if torch.cuda.is_available():
        # PyTorch 2.0/CUDA 11.8's efficient-SDP backward kernel can issue an
        # illegal instruction on H200. The math kernel is equivalent and is
        # applied identically to both frozen-backbone comparisons.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = make_datasets(args)
    train_loader = make_loader(
        train_set, args.batch_size, args.num_workers, shuffle=True, drop_last=True
    )
    val_loader = make_loader(val_set, args.eval_batch_size, args.num_workers)
    test_loader = make_loader(test_set, args.eval_batch_size, args.num_workers)

    model = GaussianCompletion(args.encoder_checkpoint, args.encoder_prefix).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    print(
        f"[{args.method}] device={device} encoder_keys={model.backbone.loaded_keys} "
        f"trainable={trainable / 1e6:.2f}M frozen={frozen / 1e6:.2f}M "
        f"train/val/test={len(train_set)}/{len(val_set)}/{len(test_set)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_val_chamfer"])

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_set.set_epoch(epoch)
        scale = learning_rate_scale(epoch, args.warmup_epochs, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.attr_weight, epoch
        )
        validation = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation,
        }
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        val_cd = validation["overall"]["chamfer"]
        if val_cd < best:
            best = val_cd
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best, args)
        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, best, args)
        print(
            f"[{args.method}] epoch={epoch + 1}/{args.epochs} "
            f"train_cd={train_metrics['chamfer']:.6f} "
            f"val_cd={val_cd:.6f} val_f={validation['overall']['fscore']:.4f} "
            f"best={best:.6f} time={train_metrics['seconds']:.0f}s",
            flush=True,
        )

    best_model, _ = load_completion(str(output_dir / "best.pth"), device)
    test_metrics = evaluate(
        best_model, test_loader, device, str(output_dir / "test_per_case.csv")
    )
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
