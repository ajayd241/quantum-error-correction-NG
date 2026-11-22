#!/usr/bin/env python3
"""Plot BP alpha tuning and vectorized batch throughput."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_study(csv_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Create one alpha tradeoff plot and one batch-throughput bar chart."""
    frame = pd.read_csv(csv_path)
    alpha_rows = frame[frame["study"] == "alpha"].sort_values("alpha")
    throughput_rows = frame[frame["study"] == "throughput"].sort_values(
        "batch_size"
    )
    if alpha_rows.empty or throughput_rows.empty:
        raise ValueError("study CSV must contain alpha and throughput rows")
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_path = out_dir / "bp_alpha_sweep.png"
    figure, ler_axis = plt.subplots(figsize=(6.5, 4.5))
    convergence_axis = ler_axis.twinx()
    ler_axis.plot(
        alpha_rows["alpha"],
        alpha_rows["ler_shot"],
        color="#d62728",
        marker="o",
        label="Logical error rate",
    )
    convergence_axis.plot(
        alpha_rows["alpha"],
        alpha_rows["convergence_rate"],
        color="#1f77b4",
        marker="s",
        label="Convergence rate",
    )
    ler_axis.set_xlabel("Min-sum normalization, alpha")
    ler_axis.set_ylabel("Logical error rate per shot", color="#d62728")
    convergence_axis.set_ylabel("Convergence rate", color="#1f77b4")
    ler_axis.set_title("BP normalization tradeoff")
    ler_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(alpha_path, dpi=200)
    plt.close(figure)

    throughput_path = out_dir / "bp_batch_throughput.png"
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    labels = [f"batch={int(value)}" for value in throughput_rows["batch_size"]]
    values = throughput_rows["shots_per_s"].to_numpy(dtype=float)
    bars = axis.bar(labels, values, color=["#9ecae1", "#2171b5"])
    speedup = values[-1] / values[0]
    axis.bar_label(bars, fmt="%.0f shots/s", padding=3)
    axis.set_ylabel("Decoded shots per second")
    axis.set_title(f"BP vectorization: {speedup:.1f}x throughput")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(throughput_path, dpi=200)
    plt.close(figure)
    return alpha_path, throughput_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figs"),
    )
    args = parser.parse_args()
    outputs = plot_study(args.csv, args.out_dir)
    for output in outputs:
        print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
