#!/usr/bin/env python3
"""Run reproducible decoder sweeps and multiprocessing benchmarks."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qecdec.harness import sweep

DEFAULT_PROBABILITIES = np.geomspace(0.002, 0.015, 8)
DEFAULT_DECODERS = ["pymatching"]
DEFAULT_DISTANCES = [3, 5, 7]
ALLOWED_CONFIG_KEYS = {
    "decoders",
    "distances",
    "probabilities",
    "shots",
    "shots_by_decoder",
    "seed",
    "workers",
    "out",
    "decoder_options",
    "parallel_benchmark",
}


def _comma_separated_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _comma_separated_ints(value: str) -> list[int]:
    try:
        return [int(item) for item in _comma_separated_strings(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _comma_separated_floats(value: str) -> list[float]:
    try:
        return [float(item) for item in _comma_separated_strings(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def load_config(path: Path) -> dict[str, Any]:
    """Load a dependency-free JSON object, which is also valid YAML 1.2."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"config {path} must contain one object")
    unknown = set(value) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return value


def _decoder_options(value: object) -> dict[str, dict[str, object]]:
    """Validate decoder-specific keyword arguments from a config object."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("decoder_options must be an object")
    normalized: dict[str, dict[str, object]] = {}
    for decoder, options in value.items():
        if not isinstance(decoder, str) or not isinstance(options, dict):
            raise ValueError("decoder_options must map names to objects")
        normalized[decoder] = dict(options)
    return normalized


def _shots(
    config: Mapping[str, Any], cli_shots: int | None
) -> int | dict[str, int]:
    """Resolve either one shot count or per-decoder shot counts."""
    if cli_shots is not None:
        return cli_shots
    if "shots_by_decoder" in config:
        value = config["shots_by_decoder"]
        if not isinstance(value, dict):
            raise ValueError("shots_by_decoder must be an object")
        return {str(name): int(count) for name, count in value.items()}
    return int(config.get("shots", 20_000))


def run_parallel_benchmark(
    config: Mapping[str, Any],
    seed: int,
    decoder_options: Mapping[str, Mapping[str, object]],
    out: Path,
) -> pd.DataFrame:
    """Measure identical sweep jobs at each requested process count."""
    worker_counts = [int(value) for value in config.get("workers", [1])]
    if not worker_counts or any(value < 1 for value in worker_counts):
        raise ValueError("parallel benchmark workers must be positive")

    records: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory(prefix="qecdec-speedup-") as temp_dir:
        for workers in worker_counts:
            frame = sweep(
                decoders=[str(value) for value in config["decoders"]],
                distances=[int(value) for value in config["distances"]],
                ps=[float(value) for value in config["probabilities"]],
                shots=int(config["shots"]),
                seed=seed,
                out_csv=Path(temp_dir) / f"workers_{workers}.csv",
                workers=workers,
                decoder_options=decoder_options,
            )
            records.append(
                {
                    "workers": workers,
                    "wall_time_s": float(frame["sweep_wall_s"].iloc[0]),
                    "jobs": len(frame),
                    "shots_per_job": int(config["shots"]),
                }
            )

    result = pd.DataFrame.from_records(records).sort_values("workers")
    one_worker = result.loc[result["workers"] == 1, "wall_time_s"]
    if one_worker.empty:
        raise ValueError("parallel benchmark must include one worker")
    result["speedup"] = float(one_worker.iloc[0]) / result["wall_time_s"]
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON-syntax YAML config; explicit CLI values override it",
    )
    parser.add_argument("--decoders", type=_comma_separated_strings)
    parser.add_argument("--distances", type=_comma_separated_ints)
    parser.add_argument(
        "--ps",
        type=_comma_separated_floats,
        help="comma-separated physical error probabilities",
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="override normalization for BP and belief-find",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--benchmark-out",
        type=Path,
        default=Path("results/parallel_speedup.csv"),
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="run the configured final sweep without the worker-count study",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config(args.config) if args.config else {}
    decoders = args.decoders or config.get("decoders", DEFAULT_DECODERS)
    distances = args.distances or config.get("distances", DEFAULT_DISTANCES)
    probabilities = args.ps or config.get(
        "probabilities", DEFAULT_PROBABILITIES.tolist()
    )
    seed = args.seed if args.seed is not None else int(config.get("seed", 1))
    workers = (
        args.workers
        if args.workers is not None
        else int(config.get("workers", 1))
    )
    out = args.out or Path(config.get("out", "results/baseline_pymatching.csv"))
    decoder_options = _decoder_options(config.get("decoder_options"))
    if args.alpha is not None:
        for decoder in {"bp", "belief_find"} & set(decoders):
            decoder_options.setdefault(decoder, {})["alpha"] = args.alpha

    frame = sweep(
        decoders=[str(value) for value in decoders],
        distances=[int(value) for value in distances],
        ps=[float(value) for value in probabilities],
        shots=_shots(config, args.shots),
        seed=seed,
        out_csv=out,
        workers=workers,
        decoder_options=decoder_options,
    )
    print(
        f"Wrote {len(frame)} experiment points to {out} "
        f"in {frame['sweep_wall_s'].iloc[0]:.2f} s with {workers} worker(s)"
    )
    print(
        frame[
            ["decoder", "d", "p", "shots", "failures", "ler_shot", "seed"]
        ].to_string(index=False)
    )

    benchmark = config.get("parallel_benchmark")
    if benchmark is not None and not args.skip_benchmark:
        if not isinstance(benchmark, dict):
            raise ValueError("parallel_benchmark must be an object")
        speedup = run_parallel_benchmark(
            benchmark,
            seed=seed,
            decoder_options=decoder_options,
            out=args.benchmark_out,
        )
        print(f"Wrote worker benchmark to {args.benchmark_out}")
        print(speedup.to_string(index=False))


if __name__ == "__main__":
    main()
