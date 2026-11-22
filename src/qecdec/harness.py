"""Reproducible sampling, scoring, timing, and sweep orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import stim

from qecdec.base import Decoder
from qecdec.circuits import make_circuit
from qecdec.decoders import REGISTRY


@dataclass
class Result:
    """Accuracy and latency metrics for one decoder experiment point."""

    decoder: str
    d: int
    p: float
    rounds: int
    shots: int
    failures: int
    ler_shot: float
    ler_shot_err: float
    ler_round: float
    t_mean_us: float
    t_p50_us: float
    t_p99_us: float
    extra: dict


def sample(
    circuit: stim.Circuit,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector clicks and true logical flips from a noisy circuit."""
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    return (
        np.asarray(detectors, dtype=np.bool_),
        np.asarray(observables, dtype=np.bool_),
    )


def _time_individual_shots(
    decoder: Decoder,
    detectors: np.ndarray,
    max_shots: int = 1_000,
) -> np.ndarray:
    """Measure single-shot latency on a deterministic syndrome subsample."""
    count = min(len(detectors), max_shots)
    indices = np.linspace(0, len(detectors) - 1, num=count, dtype=np.int64)
    elapsed_us = np.empty(count, dtype=np.float64)
    for timing_index, shot_index in enumerate(indices):
        start = time.perf_counter_ns()
        decoder.decode(detectors[shot_index])
        elapsed_us[timing_index] = (time.perf_counter_ns() - start) / 1_000.0
    return elapsed_us


def evaluate(
    decoder_name: str,
    d: int,
    p: float,
    shots: int,
    seed: int,
    **decoder_kwargs: object,
) -> Result:
    """Sample and score one decoder at one surface-code operating point.

    A shot fails when any predicted logical observable differs from the
    simulator's true observable.  ``t_mean_us`` is full-batch elapsed time per
    shot (throughput); the percentiles measure calls to ``decode`` individually.
    """
    if decoder_name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"unknown decoder {decoder_name!r}; available decoders: {available}"
        )
    if seed < 0:
        raise ValueError(f"seed must be nonnegative, got {seed}")

    rounds = d
    circuit = make_circuit(d=d, p=p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True)
    decoder = REGISTRY[decoder_name].from_dem(dem, **decoder_kwargs)
    detectors, observables = sample(circuit, shots=shots, seed=seed)

    batch_start = time.perf_counter_ns()
    predictions = np.asarray(decoder.decode_batch(detectors), dtype=np.bool_)
    batch_elapsed_ns = time.perf_counter_ns() - batch_start
    if predictions.shape != observables.shape:
        raise ValueError(
            f"decoder returned shape {predictions.shape}, expected {observables.shape}"
        )
    extra = decoder.stats()

    failed = np.any(predictions != observables, axis=1)
    failures = int(np.count_nonzero(failed))
    ler_shot = failures / shots
    ler_shot_err = float(np.sqrt(ler_shot * (1.0 - ler_shot) / shots))
    ler_round = float(1.0 - (1.0 - ler_shot) ** (1.0 / rounds))

    single_shot_us = _time_individual_shots(decoder, detectors)
    return Result(
        decoder=decoder_name,
        d=d,
        p=float(p),
        rounds=rounds,
        shots=shots,
        failures=failures,
        ler_shot=ler_shot,
        ler_shot_err=ler_shot_err,
        ler_round=ler_round,
        t_mean_us=batch_elapsed_ns / shots / 1_000.0,
        t_p50_us=float(np.percentile(single_shot_us, 50)),
        t_p99_us=float(np.percentile(single_shot_us, 99)),
        extra=extra,
    )


def _point_seed(base_seed: int, distance: int, probability_index: int) -> int:
    """Derive a stable seed shared by all decoders at one sweep point."""
    sequence = np.random.SeedSequence(
        [base_seed, distance, probability_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def sweep(
    decoders: Sequence[str],
    distances: Sequence[int],
    ps: Sequence[float],
    shots: int,
    seed: int,
    out_csv: str | Path,
    **decoder_kwargs: object,
) -> pd.DataFrame:
    """Evaluate a reproducible grid and write one row per operating point."""
    records: list[dict] = []
    probabilities = [float(p) for p in ps]
    for decoder_name in decoders:
        for distance in distances:
            for probability_index, probability in enumerate(probabilities):
                point_seed = _point_seed(seed, distance, probability_index)
                result = evaluate(
                    decoder_name=decoder_name,
                    d=distance,
                    p=probability,
                    shots=shots,
                    seed=point_seed,
                    **decoder_kwargs,
                )
                record = asdict(result)
                record["seed"] = point_seed
                record["extra"] = json.dumps(
                    record["extra"],
                    sort_keys=True,
                    default=_json_default,
                )
                records.append(record)

    frame = pd.DataFrame.from_records(records)
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame

