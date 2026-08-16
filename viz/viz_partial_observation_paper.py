#!/usr/bin/env python3
"""Plot partial-observation retrieval robustness from saved frozen-encoder results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "outputs/partial_observation"
DEFAULT_OUTPUT = ROOT / "outputs/figures/partial_observation_robustness"
JEPA_COLOR = "#087E8B"
MAE_COLOR = "#D95F59"
GRID_COLOR = "#D9DEE5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "Liberation Serif",
            ],
            "font.size": 7,
            "mathtext.fontset": "stix",
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "lines.antialiased": True,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    metrics_path = args.result_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    ratios = np.asarray(metrics["protocol"]["missing_ratios"], dtype=np.float64)
    x = ratios * 100.0

    set_paper_style()
    fig, axis = plt.subplots(figsize=(3.25, 1.55))

    styles = (
        ("jepa", MAE_COLOR, "o", "Gaussian-JEPA (Ours)"),
        ("mae", JEPA_COLOR, "s", "Gaussian-MAE"),
    )
    for method, color, marker, label in styles:
        means = []
        stds = []
        for ratio in ratios:
            record = metrics["methods"][method][f"{ratio:.2f}"]
            seed_scores = np.asarray(
                [
                    seed_record["r_at_1"] * 100.0
                    for seed_record in record["per_seed"].values()
                ],
                dtype=np.float64,
            )
            means.append(record["aggregate"]["r_at_1"] * 100.0)
            stds.append(seed_scores.std(ddof=1))

        means = np.asarray(means)
        stds = np.asarray(stds)
        axis.plot(
            x,
            means,
            color=color,
            linewidth=0.9,
            marker=marker,
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        axis.fill_between(
            x,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    axis.set_xlabel("Missing groups (%)")
    axis.set_ylabel(r"Retrieval R@1 (%) $\uparrow$")
    axis.set_xticks(x)
    axis.set_xlim(-2, 87)
    axis.set_ylim(0, 100)
    axis.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.45,
        borderaxespad=0.3,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.28, top=0.98)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        args.output.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    print("Saved", args.output.with_suffix(".pdf"))
    print("Saved", args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
