#!/usr/bin/env python3
"""Plot BP convergence rate against physical error probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_convergence(csv_path: Path, out: Path) -> Path:
    """Plot the convergence fraction stored in each BP sweep row."""
    frame = pd.read_csv(csv_path)
    bp = frame[frame["decoder"] == "bp"].copy()
    if bp.empty:
        raise ValueError(f"no bp rows found in {csv_path}")
    bp["convergence_rate"] = [
        float(json.loads(value)["convergence_rate"])
        for value in bp["extra"].astype(str)
    ]

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for distance, points in bp.groupby("d"):
        points = points.sort_values("p")
        axis.plot(
            points["p"],
            points["convergence_rate"],
            marker="o",
            linewidth=1.8,
            label=f"d={int(distance)}",
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Physical error probability, p")
    axis.set_ylabel("Fraction converged by 30 iterations")
    axis.set_title("Standalone BP convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figs/bp_convergence.png"),
    )
    args = parser.parse_args()
    output = plot_convergence(args.csv, args.out)
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
