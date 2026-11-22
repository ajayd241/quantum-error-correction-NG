"""Noisy rotated-surface-code memory circuit construction.

Repeated parity checks reveal the endpoints of physical error chains without
directly measuring the encoded logical qubit that the decoder must protect.
"""

from __future__ import annotations

import stim


def make_circuit(
    d: int,
    p: float,
    rounds: int | None = None,
    basis: str = "z",
) -> stim.Circuit:
    """Build a noisy rotated-surface-code memory experiment.

    ``after_clifford_depolarization`` applies depolarizing noise after Clifford gates.
    ``after_reset_flip_probability`` flips a qubit just after it is reset.
    ``before_measure_flip_probability`` flips a qubit just before measurement.
    ``before_round_data_depolarization`` depolarizes data qubits before each round.

    Args:
        d: Surface-code distance.
        p: Probability used for each of the four circuit-noise processes.
        rounds: Number of syndrome-measurement rounds; defaults to ``d``.
        basis: Logical memory basis, either ``"z"`` or ``"x"``.
    """
    if basis not in {"x", "z"}:
        raise ValueError(f"basis must be 'x' or 'z', got {basis!r}")
    if rounds is not None and rounds < 1:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=d if rounds is None else rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

