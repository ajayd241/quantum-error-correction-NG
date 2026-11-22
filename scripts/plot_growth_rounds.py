#!/usr/bin/env python3
"""Plot Union-Find growth-round distributions against code distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _distribution(extra_json: str) -> np.ndarray:
    """Expand a compact growth-round histogram into its sampled values."""
    extra = json.loads(extra_json)
    histogram = extra.get("growth_rounds")
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError("row has no growth_rounds histogram")
    values = np.asarray([int(value) for value in histogram], dtype=np.int32)
    counts = np.asarray(list(histogram.values()), dtype=np.int32)
    return np.repeat(values, counts)


def plot_growth_rounds(
    csv_path: Path,
    decoder: str,
    target_probability: float,
    out: Path,
) -> Path:
    """Plot exact per-shot distributions at the grid point nearest target p."""
    frame = pd.read_csv(csv_path)
    decoder_rows = frame[frame["decoder"] == decoder]
    if decoder_rows.empty:
        raise ValueError(f"decoder {decoder!r} is absent from {csv_path}")
    probabilities = decoder_rows["p"].unique()
    probability = float(
        probabilities[np.argmin(np.abs(probabilities - target_probability))]
    )
    selected = decoder_rows[decoder_rows["p"] == probability].sort_values("d")
    if selected["d"].duplicated().any():
        raise ValueError("expected one row per distance at the selected p")

    distances = selected["d"].to_numpy(dtype=int)
    distributions = [
        _distribution(extra_json)
        for extra_json in selected["extra"].astype(str)
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    violin = axis.violinplot(
        distributions,
        positions=distances,
        widths=0.9,
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#9467bd")
        body.set_edgecolor("#4b256d")
        body.set_alpha(0.75)
    axis.set_xticks(distances)
    axis.set_xlabel("Code distance, d")
    axis.set_ylabel("Cluster-growth rounds per shot")
    axis.set_title(f"{decoder} growth depth at p={probability:.4g}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200)
    plt.close(figure)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--decoder", default="uf")
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figs/uf_growth_rounds.png"),
    )
    args = parser.parse_args()
    output = plot_growth_rounds(
        args.csv,
        args.decoder,
        args.p,
        args.out,
    )
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
