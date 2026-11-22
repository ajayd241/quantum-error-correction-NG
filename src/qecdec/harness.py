"""Reproducible syndrome sampling, logical scoring, and sweep orchestration.

Every decoder sees the same seeded detector events and succeeds only when its
predicted logical-observable flips match the simulator's hidden truth.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class _SweepJob:
    """Serializable description of one independent benchmark point."""

    decoder: str
    distance: int
    probability: float
    shots: int
    seed: int
    decoder_kwargs: dict[str, object]


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


def _point_seed(base_seed: int, distance: int, probability: float) -> int:
    """Derive a stable seed shared by all decoders at one sweep point."""
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


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


def _evaluate_job(job: _SweepJob) -> dict[str, object]:
    """Evaluate one process-safe sweep job and serialize its result."""
    result = evaluate(
        decoder_name=job.decoder,
        d=job.distance,
        p=job.probability,
        shots=job.shots,
        seed=job.seed,
        **job.decoder_kwargs,
    )
    record = asdict(result)
    record["seed"] = job.seed
    record["extra"] = json.dumps(
        record["extra"],
        sort_keys=True,
        default=_json_default,
    )
    return record


def sweep(
    decoders: Sequence[str],
    distances: Sequence[int],
    ps: Sequence[float],
    shots: int | Mapping[str, int],
    seed: int,
    out_csv: str | Path,
    workers: int = 1,
    decoder_options: Mapping[str, Mapping[str, object]] | None = None,
    **decoder_kwargs: object,
) -> pd.DataFrame:
    """Evaluate a reproducible grid and write one row per operating point.

    Jobs are independent ``(decoder, d, p)`` points.  With more than one
    worker they run in separate processes, which models shot-level parallel
    throughput while keeping every point and its seed reproducible.

    ``shots`` may be one count for the full grid or a decoder-to-count map.
    ``decoder_options`` holds decoder-specific constructor arguments; the
    legacy keyword arguments are applied to every decoder.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if seed < 0:
        raise ValueError(f"seed must be nonnegative, got {seed}")

    shot_counts: dict[str, int]
    if isinstance(shots, Mapping):
        shot_counts = {str(name): int(count) for name, count in shots.items()}
        missing_counts = set(decoders) - set(shot_counts)
        if missing_counts:
            raise ValueError(
                "missing shot counts for decoders: "
                f"{sorted(missing_counts)}"
            )
    else:
        shot_counts = {str(name): int(shots) for name in decoders}
    invalid_counts = {
        name: count for name, count in shot_counts.items() if count < 1
    }
    if invalid_counts:
        raise ValueError(f"shot counts must be positive: {invalid_counts}")

    options = decoder_options or {}
    probabilities = [float(p) for p in ps]
    jobs: list[_SweepJob] = []
    for decoder_name in decoders:
        for distance in distances:
            for probability in probabilities:
                point_seed = _point_seed(seed, distance, probability)
                point_options = dict(decoder_kwargs)
                point_options.update(options.get(decoder_name, {}))
                jobs.append(
                    _SweepJob(
                        decoder=decoder_name,
                        distance=int(distance),
                        probability=probability,
                        shots=shot_counts[decoder_name],
                        seed=point_seed,
                        decoder_kwargs=point_options,
                    )
                )

    started = time.perf_counter()
    if workers == 1:
        records = [_evaluate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            records = pool.map(_evaluate_job, jobs, chunksize=1)
    wall_time_s = time.perf_counter() - started

    frame = pd.DataFrame.from_records(records)
    frame["sweep_workers"] = workers
    frame["sweep_wall_s"] = wall_time_s
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame

