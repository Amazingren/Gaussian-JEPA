"""Select reproducible, category-diverse completion cases from test metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_csv", required=True)
    parser.add_argument("--mae_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visible_ratio", type=float, default=0.5)
    parser.add_argument("--case_seed", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=12)
    parser.add_argument("--one_per_taxonomy", action="store_true")
    return parser.parse_args()


def read_cases(path, ratio, seed):
    selected = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["visible_ratio"]) - ratio) > 1e-4:
                continue
            if int(row["case_seed"]) != seed:
                continue
            selected[row["model_id"]] = {
                "taxonomy": row["taxonomy"],
                "chamfer": float(row["chamfer"]),
                "fscore": float(row["fscore"]),
            }
    return selected


def main():
    args = parse_args()
    jepa = read_cases(args.jepa_csv, args.visible_ratio, args.case_seed)
    mae = read_cases(args.mae_csv, args.visible_ratio, args.case_seed)
    if set(jepa) != set(mae):
        missing_jepa = sorted(set(mae) - set(jepa))
        missing_mae = sorted(set(jepa) - set(mae))
        raise RuntimeError(
            f"unmatched cases: missing JEPA={missing_jepa[:3]}, "
            f"missing MAE={missing_mae[:3]}"
        )

    records = []
    for model_id in jepa:
        jepa_case = jepa[model_id]
        mae_case = mae[model_id]
        cd_gain = mae_case["chamfer"] - jepa_case["chamfer"]
        relative_cd_gain = cd_gain / max(mae_case["chamfer"], 1e-8)
        fscore_gain = jepa_case["fscore"] - mae_case["fscore"]
        records.append(
            {
                "model_id": model_id,
                "taxonomy": jepa_case["taxonomy"],
                "jepa_chamfer": jepa_case["chamfer"],
                "mae_chamfer": mae_case["chamfer"],
                "relative_cd_gain": relative_cd_gain,
                "jepa_fscore": jepa_case["fscore"],
                "mae_fscore": mae_case["fscore"],
                "fscore_gain": fscore_gain,
                "selection_score": relative_cd_gain + 0.25 * fscore_gain,
            }
        )
    records.sort(key=lambda row: row["selection_score"], reverse=True)
    if args.one_per_taxonomy:
        diverse = []
        used = set()
        for row in records:
            if row["taxonomy"] in used:
                continue
            diverse.append(row)
            used.add(row["taxonomy"])
        records = diverse
    records = records[: args.num_candidates]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "visible_ratio": args.visible_ratio,
        "case_seed": args.case_seed,
        "selection": "relative CD gain + 0.25 * absolute F-score gain",
        "one_per_taxonomy": args.one_per_taxonomy,
        "cases": records,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("Selected model IDs:")
    print(" ".join(row["model_id"] for row in records))
    for rank, row in enumerate(records, start=1):
        print(
            f"{rank:2d}. {row['model_id']} taxonomy={row['taxonomy']} "
            f"CD {row['mae_chamfer']:.5f}->{row['jepa_chamfer']:.5f} "
            f"F {row['mae_fscore']:.4f}->{row['jepa_fscore']:.4f}"
        )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
