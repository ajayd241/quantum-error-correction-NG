#!/usr/bin/env python3
"""Run the BP normalization and batch-throughput studies reproducibly."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qecdec.circuits import make_circuit
from qecdec.decoders.bp import BPDecoder
from qecdec.harness import sample

DEFAULT_ALPHAS = [0.625, 0.75, 1.0]


def _comma_separated_floats(value: str) -> list[float]:
    """Parse a nonempty comma-separated list of normalization factors."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one alpha")
    return values


def _score(
    decoder: BPDecoder,
    syndromes: np.ndarray,
    observables: np.ndarray,
) -> dict[str, float | int]:
    """Time one vectorized decode and return accuracy/convergence metrics."""
    start = time.perf_counter_ns()
    predictions = decoder.decode_batch(syndromes)
    elapsed_s = (time.perf_counter_ns() - start) / 1e9
    failures = int(
        np.count_nonzero(np.any(predictions != observables, axis=1))
    )
    stats = decoder.stats()
    return {
        "failures": failures,
        "ler_shot": failures / len(syndromes),
        "convergence_rate": float(stats["convergence_rate"]),
        "mean_iterations": float(stats["mean_iterations"]),
        "elapsed_s": elapsed_s,
        "shots_per_s": len(syndromes) / elapsed_s,
    }


def run_study(
    d: int,
    p: float,
    shots: int,
    seed: int,
    alphas: list[float],
    out: Path,
) -> pd.DataFrame:
    """Write alpha-tuning and batch-size measurements using shared shots."""
    circuit = make_circuit(d=d, p=p)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes, observables = sample(circuit, shots=shots, seed=seed)
    records: list[dict[str, object]] = []

    alpha_scores: list[dict[str, object]] = []
    for alpha in alphas:
        decoder = BPDecoder.from_dem(dem, alpha=alpha)
        score = _score(decoder, syndromes, observables)
        record: dict[str, object] = {
            "study": "alpha",
            "d": d,
            "p": p,
            "shots": shots,
            "seed": seed,
            "alpha": alpha,
            "batch_size": decoder.batch_size,
            **score,
        }
        alpha_scores.append(record)
        records.append(record)

    best = min(
        alpha_scores,
        key=lambda row: (
            int(row["failures"]),
            -float(row["convergence_rate"]),
            abs(float(row["alpha"]) - 0.75),
        ),
    )
    best_alpha = float(best["alpha"])
    for record in alpha_scores:
        record["selected_alpha"] = float(record["alpha"]) == best_alpha

    throughput_shots = min(1_000, shots)
    throughput_syndromes = syndromes[:throughput_shots]
    throughput_observables = observables[:throughput_shots]
    for batch_size in [1, throughput_shots]:
        decoder = BPDecoder.from_dem(
            dem,
            alpha=best_alpha,
            batch_size=batch_size,
        )
        decoder.decode_batch(throughput_syndromes[: min(8, throughput_shots)])
        score = _score(
            decoder,
            throughput_syndromes,
            throughput_observables,
        )
        records.append(
            {
                "study": "throughput",
                "d": d,
                "p": p,
                "shots": throughput_shots,
                "seed": seed,
                "alpha": best_alpha,
                "batch_size": batch_size,
                "selected_alpha": True,
                **score,
            }
        )

    frame = pd.DataFrame.from_records(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--shots", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_026_112_2)
    parser.add_argument(
        "--alphas",
        type=_comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/bp_study.csv"),
    )
    args = parser.parse_args()
    frame = run_study(
        d=args.d,
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        alphas=args.alphas,
        out=args.out,
    )
    print(f"Wrote {len(frame)} study rows to {args.out}")
    print(
        frame[
            [
                "study",
                "alpha",
                "batch_size",
                "failures",
                "ler_shot",
                "convergence_rate",
                "mean_iterations",
                "shots_per_s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
