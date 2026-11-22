#!/usr/bin/env python3
"""Run a reproducible decoder sweep and write its metrics to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qecdec.harness import sweep


DEFAULT_PROBABILITIES = np.geomspace(0.002, 0.015, 8)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoders", type=_comma_separated_strings, default=["pymatching"])
    parser.add_argument("--distances", type=_comma_separated_ints, default=[3, 5, 7])
    parser.add_argument(
        "--ps",
        type=_comma_separated_floats,
        default=DEFAULT_PROBABILITIES.tolist(),
        help="comma-separated physical error probabilities",
    )
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/baseline_pymatching.csv"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    frame = sweep(
        decoders=args.decoders,
        distances=args.distances,
        ps=args.ps,
        shots=args.shots,
        seed=args.seed,
        out_csv=args.out,
    )
    print(f"Wrote {len(frame)} experiment points to {args.out}")
    print(
        frame[
            ["decoder", "d", "p", "shots", "failures", "ler_shot", "seed"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

