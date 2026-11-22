#!/usr/bin/env python3
"""Run the reproducible belief-find iteration-budget and fast-path study."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qecdec.circuits import make_circuit
from qecdec.decoders.belief_find import BeliefFindDecoder
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.decoders.union_find import UnionFindDecoder
from qecdec.harness import sample

DEFAULT_BUDGETS = [0, 1, 2, 4, 8, 16, 32]
DEFAULT_DISTANCES = [3, 5, 7]
DEFAULT_PROBABILITIES = [0.0035, 0.00625]


def _comma_separated_ints(value: str) -> list[int]:
    """Parse a nonempty comma-separated integer list."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _comma_separated_floats(value: str) -> list[float]:
    """Parse a nonempty comma-separated float list."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one probability")
    return values


def _point_seed(base_seed: int, distance: int, probability: float) -> int:
    """Derive a stable seed that is unchanged by grid ordering."""
    probability_bits = int(np.float64(probability).view(np.uint64))
    sequence = np.random.SeedSequence(
        [
            base_seed & 0xFFFFFFFF,
            (base_seed >> 32) & 0xFFFFFFFF,
            distance & 0xFFFFFFFF,
            probability_bits & 0xFFFFFFFF,
            probability_bits >> 32,
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _measure(
    decoder: object,
    syndromes: np.ndarray,
    observables: np.ndarray,
    latency_shots: int,
) -> tuple[np.ndarray, float, float, dict]:
    """Measure batch throughput and deterministic single-shot p99 latency."""
    start = time.perf_counter_ns()
    predictions = np.asarray(decoder.decode_batch(syndromes), dtype=np.bool_)
    elapsed_ns = time.perf_counter_ns() - start
    stats = dict(decoder.stats())

    count = min(latency_shots, len(syndromes))
    indices = np.linspace(0, len(syndromes) - 1, num=count, dtype=np.int64)
    individual_us = np.empty(count, dtype=np.float64)
    for timing_index, shot_index in enumerate(indices):
        start = time.perf_counter_ns()
        decoder.decode(syndromes[shot_index])
        individual_us[timing_index] = (
            time.perf_counter_ns() - start
        ) / 1_000.0
    return (
        predictions,
        elapsed_ns / len(syndromes) / 1_000.0,
        float(np.percentile(individual_us, 99)),
        stats,
    )


def _record(
    decoder_name: str,
    predictions: np.ndarray,
    observables: np.ndarray,
    d: int,
    p: float,
    seed: int,
    t_mean_us: float,
    t_p99_us: float,
    stats: dict,
    bp_iters: int | None,
    fast_path: bool | None,
) -> dict[str, object]:
    """Create one tidy study row from predictions and timing measurements."""
    shots = len(observables)
    failures = int(np.count_nonzero(np.any(predictions != observables, axis=1)))
    ler = failures / shots
    return {
        "decoder": decoder_name,
        "d": d,
        "p": p,
        "shots": shots,
        "seed": seed,
        "bp_iters": bp_iters,
        "fast_path": fast_path,
        "failures": failures,
        "ler_shot": ler,
        "ler_shot_err": float(np.sqrt(ler * (1.0 - ler) / shots)),
        "t_mean_us": t_mean_us,
        "t_p99_us": t_p99_us,
        "fast_path_fraction": stats.get("fast_path_fraction", np.nan),
        "mean_bp_iterations": stats.get("mean_bp_iterations", np.nan),
        "alpha": stats.get("alpha", np.nan),
    }


def run_budget_study(
    distances: list[int],
    probabilities: list[float],
    budgets: list[int],
    shots: int,
    latency_shots: int,
    seed: int,
    alpha: float,
    out: Path,
) -> pd.DataFrame:
    """Run shared-shot baselines and belief-find configurations."""
    records: list[dict[str, object]] = []
    low_probability = min(probabilities)
    for distance in distances:
        for probability in probabilities:
            point_seed = _point_seed(seed, distance, probability)
            circuit = make_circuit(distance, probability)
            dem = circuit.detector_error_model(decompose_errors=True)
            syndromes, observables = sample(circuit, shots, point_seed)

            for name, decoder in [
                ("pymatching", PyMatchingDecoder.from_dem(dem)),
                ("uf", UnionFindDecoder.from_dem(dem)),
            ]:
                predictions, mean_us, p99_us, stats = _measure(
                    decoder,
                    syndromes,
                    observables,
                    latency_shots,
                )
                row = _record(
                    name,
                    predictions,
                    observables,
                    distance,
                    probability,
                    point_seed,
                    mean_us,
                    p99_us,
                    stats,
                    None,
                    None,
                )
                row["regime"] = (
                    "0.5x threshold"
                    if probability == low_probability
                    else "0.9x threshold"
                )
                records.append(row)

            for budget in budgets:
                for use_fast_path in [False, True]:
                    decoder = BeliefFindDecoder.from_dem(
                        dem,
                        bp_iters=budget,
                        fast_path=use_fast_path,
                        alpha=alpha,
                    )
                    predictions, mean_us, p99_us, stats = _measure(
                        decoder,
                        syndromes,
                        observables,
                        latency_shots,
                    )
                    row = _record(
                        "belief_find",
                        predictions,
                        observables,
                        distance,
                        probability,
                        point_seed,
                        mean_us,
                        p99_us,
                        stats,
                        budget,
                        use_fast_path,
                    )
                    row["regime"] = (
                        "0.5x threshold"
                        if probability == low_probability
                        else "0.9x threshold"
                    )
                    records.append(row)

            frame = pd.DataFrame.from_records(records)
            out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(out, index=False)
            print(
                f"Completed d={distance}, p={probability:g}; "
                f"wrote {len(frame)} rows to {out}",
                flush=True,
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distances",
        type=_comma_separated_ints,
        default=DEFAULT_DISTANCES,
    )
    parser.add_argument(
        "--ps",
        type=_comma_separated_floats,
        default=DEFAULT_PROBABILITIES,
    )
    parser.add_argument(
        "--budgets",
        type=_comma_separated_ints,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--latency-shots", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.625)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/budget.csv"),
    )
    args = parser.parse_args()
    if args.shots < 1 or args.latency_shots < 1:
        parser.error("shots and latency-shots must be positive")
    frame = run_budget_study(
        distances=args.distances,
        probabilities=args.ps,
        budgets=args.budgets,
        shots=args.shots,
        latency_shots=args.latency_shots,
        seed=args.seed,
        alpha=args.alpha,
        out=args.out,
    )
    print(f"Wrote {len(frame)} total rows to {args.out}")


if __name__ == "__main__":
    main()
