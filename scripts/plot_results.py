#!/usr/bin/env python3
"""Generate the complete, consistently styled final benchmark figure suite."""

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
    "t_p50_us",
    "t_p99_us",
}
DECODER_ORDER = [
    "pymatching",
    "uf_unweighted",
    "uf",
    "bp",
    "belief_find",
    "mwpm_nx",
]
DECODER_LABELS = {
    "pymatching": "PyMatching",
    "uf_unweighted": "UF (unweighted)",
    "uf": "UF (weighted)",
    "bp": "BP",
    "belief_find": "Belief-find",
    "mwpm_nx": "MWPM (NetworkX)",
}
DECODER_COLORS = {
    "pymatching": "#222222",
    "uf_unweighted": "#9c755f",
    "uf": "#4e79a7",
    "bp": "#e15759",
    "belief_find": "#59a14f",
    "mwpm_nx": "#b07aa1",
}
DISTANCE_STYLES = {
    3: ("o", "-"),
    5: ("s", "--"),
    7: ("^", "-."),
}


def _validate(frame: pd.DataFrame) -> None:
    """Reject incomplete sweep tables before producing misleading figures."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required CSV columns: {sorted(missing)}")


def _decoder_sort_key(decoder: str) -> tuple[int, str]:
    try:
        return DECODER_ORDER.index(decoder), decoder
    except ValueError:
        return len(DECODER_ORDER), decoder


def _decoders(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (str(value) for value in frame["decoder"].unique()),
        key=_decoder_sort_key,
    )


def _color(decoder: str) -> str:
    return DECODER_COLORS.get(decoder, "#f28e2b")


def _label(decoder: str) -> str:
    return DECODER_LABELS.get(decoder, decoder)


def _plottable_ler(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return positive log-axis values and asymmetric lower/upper errors."""
    y = points["ler_shot"].to_numpy(dtype=float)
    shots = points["shots"].to_numpy(dtype=float)
    plotted_y = np.where(y > 0.0, y, 0.5 / shots)
    error = points["ler_shot_err"].to_numpy(dtype=float)
    lower_error = np.minimum(error, plotted_y * (1.0 - 1e-12))
    return plotted_y, np.vstack([lower_error, error])


def _finish_ler_axis(axis: plt.Axes, title: str) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Physical error probability, p")
    axis.set_ylabel("Logical error rate per shot")
    axis.set_title(title)
    axis.grid(which="both", alpha=0.25)
    axis.legend(fontsize=8)


def plot_decoder_ler(
    frame: pd.DataFrame, decoder: str, out_dir: Path
) -> Path:
    """Plot one decoder's logical error curves, one line per code distance."""
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    selected = frame[frame["decoder"] == decoder]
    for distance in sorted(int(value) for value in selected["d"].unique()):
        points = selected[selected["d"] == distance].sort_values("p")
        marker, linestyle = DISTANCE_STYLES.get(distance, ("o", "-"))
        y, error = _plottable_ler(points)
        axis.errorbar(
            points["p"],
            y,
            yerr=error,
            color=_color(decoder),
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.7,
            capsize=2.5,
            label=f"d={distance}",
        )
    _finish_ler_axis(axis, f"{_label(decoder)} logical error rate")
    figure.tight_layout()
    output = out_dir / f"ler_vs_p_{decoder}.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_d5_comparison(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Compare every decoder at distance five on one log-log plot."""
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    for decoder in _decoders(frame):
        points = frame[
            (frame["decoder"] == decoder) & (frame["d"] == 5)
        ].sort_values("p")
        if points.empty:
            continue
        y, error = _plottable_ler(points)
        axis.errorbar(
            points["p"],
            y,
            yerr=error,
            color=_color(decoder),
            marker="o",
            markersize=4,
            linewidth=1.7,
            capsize=2.5,
            label=_label(decoder),
        )
    _finish_ler_axis(axis, "Decoder comparison at d=5")
    figure.tight_layout()
    output = out_dir / "ler_vs_p_d5_all.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def estimate_threshold(frame: pd.DataFrame, decoder: str) -> float | None:
    """Estimate the d=5/d=7 crossing by linear interpolation in log-log space."""
    selected = frame[frame["decoder"] == decoder]
    d5 = selected[selected["d"] == 5][["p", "ler_shot"]].rename(
        columns={"ler_shot": "ler_5"}
    )
    d7 = selected[selected["d"] == 7][["p", "ler_shot"]].rename(
        columns={"ler_shot": "ler_7"}
    )
    paired = d5.merge(d7, on="p").sort_values("p")
    paired = paired[(paired["ler_5"] > 0.0) & (paired["ler_7"] > 0.0)]
    if len(paired) < 2:
        return None

    x = np.log(paired["p"].to_numpy(dtype=float))
    difference = np.log(paired["ler_5"].to_numpy(dtype=float)) - np.log(
        paired["ler_7"].to_numpy(dtype=float)
    )
    for index in range(len(difference) - 1):
        left = difference[index]
        right = difference[index + 1]
        if left == 0.0:
            return float(np.exp(x[index]))
        if left < 0.0 or right > 0.0:
            continue
        fraction = -left / (right - left)
        return float(np.exp(x[index] + fraction * (x[index + 1] - x[index])))
    if difference[-1] == 0.0:
        return float(np.exp(x[-1]))
    return None


def plot_thresholds(frame: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Write threshold estimates as both a figure and a machine-readable table."""
    records = [
        {"decoder": decoder, "threshold": estimate_threshold(frame, decoder)}
        for decoder in _decoders(frame)
    ]
    estimates = pd.DataFrame.from_records(records)
    table_path = out_dir.parent / "threshold_estimates.csv"
    estimates.to_csv(table_path, index=False)

    available = estimates.dropna(subset=["threshold"])
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    if available.empty:
        axis.text(
            0.5,
            0.5,
            "No d=5/d=7 crossing in the sampled range",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        decoders = available["decoder"].astype(str).tolist()
        values = 100.0 * available["threshold"].to_numpy(dtype=float)
        bars = axis.bar(
            [_label(decoder) for decoder in decoders],
            values,
            color=[_color(decoder) for decoder in decoders],
        )
        axis.bar_label(bars, fmt="%.3f%%", padding=3, fontsize=8)
        axis.set_ylabel("Estimated threshold (%)")
        axis.set_title("d=5/d=7 log-log crossing estimates")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure_path = out_dir / "threshold_estimates.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    return figure_path, table_path


def _nearest_operating_points(
    frame: pd.DataFrame, target_p: float = 0.005
) -> pd.DataFrame:
    """Select one d=5 row nearest the requested physical error rate."""
    rows: list[pd.Series] = []
    for decoder in _decoders(frame):
        points = frame[
            (frame["decoder"] == decoder) & (frame["d"] == 5)
        ]
        if points.empty:
            continue
        nearest_index = (points["p"] - target_p).abs().idxmin()
        rows.append(points.loc[nearest_index])
    return pd.DataFrame(rows)


def plot_latency(frame: pd.DataFrame, out_dir: Path) -> Path:
    """Plot single-shot median and tail latency near d=5, p=0.005."""
    points = _nearest_operating_points(frame)
    decoders = points["decoder"].astype(str).tolist()
    x = np.arange(len(points), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(
        x - width / 2,
        points["t_p50_us"],
        width,
        color=[_color(decoder) for decoder in decoders],
        alpha=0.72,
        label="p50",
    )
    axis.bar(
        x + width / 2,
        points["t_p99_us"],
        width,
        color=[_color(decoder) for decoder in decoders],
        hatch="//",
        label="p99",
    )
    axis.set_yscale("log")
    axis.set_xticks(x, [_label(decoder) for decoder in decoders], rotation=20)
    axis.set_ylabel("Single-shot decode latency (µs, log scale)")
    axis.set_title("Latency at d=5, p=0.005")
    axis.grid(axis="y", which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = out_dir / "latency.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_parallel_speedup(csv_path: Path, out_dir: Path) -> Path:
    """Plot measured end-to-end sweep speedup against ideal linear scaling."""
    frame = pd.read_csv(csv_path).sort_values("workers")
    required = {"workers", "wall_time_s", "speedup"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing speedup columns: {sorted(missing)}")
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(
        frame["workers"],
        frame["speedup"],
        color="#4e79a7",
        marker="o",
        linewidth=1.8,
        label="Measured",
    )
    axis.plot(
        frame["workers"],
        frame["workers"],
        color="#888888",
        linestyle="--",
        label="Ideal",
    )
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Speedup vs one worker")
    axis.set_title("Parallel sweep scaling")
    axis.set_xticks(frame["workers"])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = out_dir / "parallel_speedup.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_all_results(
    csv_path: Path,
    out_dir: Path,
    parallel_csv: Path | None = None,
) -> list[Path]:
    """Generate every final-sweep figure and return the written paths."""
    frame = pd.read_csv(csv_path)
    _validate(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_decoder_ler(frame, decoder, out_dir)
        for decoder in _decoders(frame)
    ]
    outputs.append(plot_d5_comparison(frame, out_dir))
    threshold_figure, threshold_table = plot_thresholds(frame, out_dir)
    outputs.extend([threshold_figure, threshold_table])
    outputs.append(plot_latency(frame, out_dir))
    if parallel_csv is not None:
        outputs.append(plot_parallel_speedup(parallel_csv, out_dir))
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figs"))
    parser.add_argument("--parallel-csv", type=Path)
    parser.add_argument("--budget-csv", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    outputs = plot_all_results(args.csv, args.out, args.parallel_csv)
    if args.budget_csv is not None:
        from plot_budget_study import plot_budget_study

        outputs.extend(plot_budget_study(args.budget_csv, args.out))
    print("Wrote figures:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
