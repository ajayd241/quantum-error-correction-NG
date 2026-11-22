#!/usr/bin/env python3
"""Plot logical error rate against physical error rate from a sweep CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "decoder",
    "d",
    "p",
    "shots",
    "ler_shot",
    "ler_shot_err",
}
LINE_STYLES = ["-", "--", "-.", ":"]


def _output_path(csv_path: Path, requested: Path) -> Path:
    if requested.suffix.lower() == ".png":
        return requested
    return requested / f"{csv_path.stem}.png"


def plot_results(csv_path: Path, out: Path) -> Path:
    """Create the common log-log comparison plot and return its path."""
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required CSV columns: {sorted(missing)}")

    distances = sorted(int(distance) for distance in frame["d"].unique())
    decoders = sorted(str(decoder) for decoder in frame["decoder"].unique())
    colors = plt.get_cmap("tab10")
    color_by_distance = {
        distance: colors(index % 10) for index, distance in enumerate(distances)
    }
    style_by_decoder = {
        decoder: LINE_STYLES[index % len(LINE_STYLES)]
        for index, decoder in enumerate(decoders)
    }

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for decoder in decoders:
        for distance in distances:
            points = frame[
                (frame["decoder"] == decoder) & (frame["d"] == distance)
            ].sort_values("p")
            if points.empty:
                continue

            y = points["ler_shot"].to_numpy(dtype=float)
            shots = points["shots"].to_numpy(dtype=float)
            plotted_y = np.where(y > 0.0, y, 0.5 / shots)
            error = points["ler_shot_err"].to_numpy(dtype=float)
            lower_error = np.minimum(error, plotted_y * (1.0 - 1e-12))
            asymmetric_error = np.vstack([lower_error, error])
            axis.errorbar(
                points["p"],
                plotted_y,
                yerr=asymmetric_error,
                color=color_by_distance[distance],
                linestyle=style_by_decoder[decoder],
                marker="o",
                markersize=4,
                linewidth=1.6,
                capsize=2.5,
                label=f"{decoder}, d={distance}",
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error probability, p")
    axis.set_ylabel("Logical error rate per shot")
    axis.set_title("Surface-code decoder benchmark")
    axis.grid(which="both", alpha=0.25)
    axis.legend(fontsize=8, ncols=max(1, len(decoders)))
    figure.tight_layout()

    output = _output_path(csv_path, out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figs"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = plot_results(args.csv, args.out)
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()

