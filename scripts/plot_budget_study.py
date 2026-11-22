#!/usr/bin/env python3
"""Plot belief-find accuracy, tail latency, and fast-path utilization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _panels(frame: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Return stable distance and probability panel orderings."""
    return (
        sorted(int(value) for value in frame["d"].unique()),
        sorted(float(value) for value in frame["p"].unique()),
    )


def plot_accuracy(frame: pd.DataFrame, out: Path) -> Path:
    """Plot LER versus BP budget normalized to weighted UF."""
    distances, probabilities = _panels(frame)
    figure, axes = plt.subplots(
        len(distances),
        len(probabilities),
        figsize=(10.5, 8.0),
        sharex=True,
        squeeze=False,
    )
    for row, distance in enumerate(distances):
        for column, probability in enumerate(probabilities):
            axis = axes[row, column]
            point = frame[(frame["d"] == distance) & (frame["p"] == probability)]
            uf = point[point["decoder"] == "uf"].iloc[0]
            pymatching = point[point["decoder"] == "pymatching"].iloc[0]
            denominator = max(float(uf["ler_shot"]), 0.5 / float(uf["shots"]))
            hybrid = point[
                (point["decoder"] == "belief_find")
                & (point["fast_path"].astype(str) == "True")
            ].sort_values("bp_iters")
            values = hybrid["ler_shot"].to_numpy(dtype=float) / denominator
            errors = hybrid["ler_shot_err"].to_numpy(dtype=float) / denominator
            axis.errorbar(
                hybrid["bp_iters"],
                values,
                yerr=errors,
                color="#1f77b4",
                marker="o",
                capsize=2,
                label="Belief-find",
            )
            axis.axhline(1.0, color="#555555", linewidth=1.2, label="UF")
            pm_ratio = max(
                float(pymatching["ler_shot"]),
                0.5 / float(pymatching["shots"]),
            ) / denominator
            axis.axhline(
                pm_ratio,
                color="#d62728",
                linestyle="--",
                linewidth=1.4,
                label="PyMatching",
            )
            axis.set_xscale("symlog", linthresh=1.0, base=2)
            axis.grid(alpha=0.25)
            axis.set_title(f"d={distance}, p={probability:g}")
            if column == 0:
                axis.set_ylabel("LER / weighted-UF LER")
            if row == len(distances) - 1:
                axis.set_xlabel("BP iteration budget, k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=3,
    )
    figure.suptitle("Belief-find accuracy versus BP budget", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def plot_latency(frame: pd.DataFrame, out: Path) -> Path:
    """Compare p99 latency with and without the BP-converged fast path."""
    distances, probabilities = _panels(frame)
    figure, axes = plt.subplots(
        len(distances),
        len(probabilities),
        figsize=(10.5, 8.0),
        sharex=True,
        squeeze=False,
    )
    for row, distance in enumerate(distances):
        for column, probability in enumerate(probabilities):
            axis = axes[row, column]
            point = frame[
                (frame["decoder"] == "belief_find")
                & (frame["d"] == distance)
                & (frame["p"] == probability)
            ]
            for fast_path, label, color in [
                (False, "without fast path", "#9ecae1"),
                (True, "with fast path", "#08519c"),
            ]:
                rows = point[
                    point["fast_path"].astype(str) == str(fast_path)
                ].sort_values("bp_iters")
                axis.plot(
                    rows["bp_iters"],
                    rows["t_p99_us"],
                    marker="o",
                    color=color,
                    label=label,
                )
            axis.set_xscale("symlog", linthresh=1.0, base=2)
            axis.grid(alpha=0.25)
            axis.set_title(f"d={distance}, p={probability:g}")
            if column == 0:
                axis.set_ylabel("p99 latency (µs)")
            if row == len(distances) - 1:
                axis.set_xlabel("BP iteration budget, k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncols=2,
    )
    figure.suptitle("Belief-find tail latency", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def plot_fast_path(frame: pd.DataFrame, out: Path) -> Path:
    """Plot the maximum-budget BP-only completion fraction versus noise."""
    hybrid = frame[
        (frame["decoder"] == "belief_find")
        & (frame["fast_path"].astype(str) == "True")
    ].copy()
    max_budget = int(hybrid["bp_iters"].max())
    hybrid = hybrid[hybrid["bp_iters"] == max_budget]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for distance, rows in hybrid.groupby("d"):
        rows = rows.sort_values("p")
        axis.plot(
            rows["p"],
            rows["fast_path_fraction"],
            marker="o",
            label=f"d={int(distance)}",
        )
    axis.set_xlabel("Physical error probability, p")
    axis.set_ylabel("BP-converged fast-path fraction")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title(f"Fast-path utilization at k={max_budget}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def plot_budget_study(csv_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    """Generate all three required iteration-budget study figures."""
    frame = pd.read_csv(csv_path)
    outputs = (
        plot_accuracy(frame, out_dir / "budget_study_accuracy.png"),
        plot_latency(frame, out_dir / "budget_study_latency.png"),
        plot_fast_path(frame, out_dir / "budget_study_fast_path.png"),
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figs"))
    args = parser.parse_args()
    for output in plot_budget_study(args.csv, args.out_dir):
        print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
