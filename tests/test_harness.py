"""Tests for reproducible sampling and decoder evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qecdec.circuits import make_circuit
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.harness import _point_seed, evaluate, sample, sweep


def test_zero_syndrome_has_zero_prediction() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    decoder = PyMatchingDecoder.from_dem(dem)

    prediction = decoder.decode(np.zeros(dem.num_detectors, dtype=np.bool_))

    np.testing.assert_array_equal(
        prediction,
        np.zeros(dem.num_observables, dtype=np.bool_),
    )


def test_sampling_with_fixed_seed_is_identical() -> None:
    circuit = make_circuit(3, 0.005)

    first_detectors, first_observables = sample(circuit, shots=100, seed=1234)
    second_detectors, second_observables = sample(
        circuit, shots=100, seed=1234
    )

    np.testing.assert_array_equal(first_detectors, second_detectors)
    np.testing.assert_array_equal(first_observables, second_observables)


def test_evaluate_with_fixed_seed_has_same_failures() -> None:
    first = evaluate("pymatching", d=3, p=0.005, shots=2_000, seed=99)
    second = evaluate("pymatching", d=3, p=0.005, shots=2_000, seed=99)

    assert first.failures == second.failures
    assert first.ler_shot == second.ler_shot
    assert first.ler_round == second.ler_round
    assert first.ler_shot_err == second.ler_shot_err


def test_sweep_writes_csv_with_recorded_seed(tmp_path) -> None:
    output = tmp_path / "nested" / "sweep.csv"

    frame = sweep(
        decoders=["pymatching"],
        distances=[3],
        ps=[0.005],
        shots=100,
        seed=7,
        out_csv=output,
    )

    assert output.is_file()
    from_disk = pd.read_csv(output)
    assert len(frame) == len(from_disk) == 1
    assert frame.loc[0, "seed"] == from_disk.loc[0, "seed"]
    assert from_disk.loc[0, "decoder"] == "pymatching"


def test_point_seed_depends_on_probability_not_grid_position() -> None:
    low_first = {
        probability: _point_seed(7, 5, probability)
        for probability in [0.002, 0.005]
    }
    high_first = {
        probability: _point_seed(7, 5, probability)
        for probability in [0.005, 0.002]
    }

    assert low_first == high_first
    assert low_first[0.002] != low_first[0.005]


@pytest.mark.slow
def test_lower_logical_error_rate_at_larger_distance_below_threshold() -> None:
    distance_three = evaluate(
        "pymatching", d=3, p=0.002, shots=50_000, seed=2026
    )
    distance_five = evaluate(
        "pymatching", d=5, p=0.002, shots=50_000, seed=2026
    )

    assert distance_five.ler_shot < distance_three.ler_shot

