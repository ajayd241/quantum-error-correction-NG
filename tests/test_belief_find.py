"""Correctness and statistical checks for BP-guided Union-Find."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qecdec.circuits import make_circuit
from qecdec.decoders.belief_find import BeliefFindDecoder
from qecdec.decoders.union_find import UnionFindDecoder
from qecdec.dem_graph import build_from_dem
from qecdec.harness import sample


def test_edge_probability_combine_matches_xor_rule() -> None:
    one_mechanism = BeliefFindDecoder.from_dem(
        stim.DetectorErrorModel("error(0.1) D0")
    )
    two_mechanisms = BeliefFindDecoder.from_dem(
        stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D0")
    )

    np.testing.assert_allclose(
        one_mechanism.edge_probabilities(np.asarray([0.23])),
        [0.23],
        rtol=0.0,
        atol=1e-12,
    )
    first, second = 0.23, 0.31
    np.testing.assert_allclose(
        two_mechanisms.edge_probabilities(np.asarray([first, second])),
        [first * (1.0 - second) + second * (1.0 - first)],
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_bp_iterations_matches_weighted_union_find_bit_for_bit() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes, _ = sample(circuit, shots=2_000, seed=301)
    union_find = UnionFindDecoder.from_dem(dem)
    belief_find = BeliefFindDecoder.from_dem(
        dem,
        bp_iters=0,
        fast_path=False,
    )

    uf_predictions = union_find.decode_batch(syndromes)
    bf_predictions = belief_find.decode_batch(syndromes)

    np.testing.assert_array_equal(bf_predictions, uf_predictions)


def test_belief_find_corrects_every_distance_three_single_fault() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    _, matrices = build_from_dem(dem)
    decoder = BeliefFindDecoder.from_dem(dem)
    syndromes = matrices.H.T.toarray().astype(np.bool_)
    expected = matrices.L.T.toarray().astype(np.bool_)

    predictions = decoder.decode_batch(syndromes)

    np.testing.assert_array_equal(predictions, expected)


@pytest.mark.slow
def test_belief_find_beats_union_find_near_threshold() -> None:
    circuit = make_circuit(5, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes, observables = sample(circuit, shots=5_000, seed=20_261_123)
    union_find = UnionFindDecoder.from_dem(dem)
    belief_find = BeliefFindDecoder.from_dem(dem)

    uf_predictions = union_find.decode_batch(syndromes)
    bf_predictions = belief_find.decode_batch(syndromes)
    uf_failures = int(
        np.count_nonzero(np.any(uf_predictions != observables, axis=1))
    )
    bf_failures = int(
        np.count_nonzero(np.any(bf_predictions != observables, axis=1))
    )
    shots = len(syndromes)
    uf_ler = uf_failures / shots
    bf_ler = bf_failures / shots
    uf_error = np.sqrt(uf_ler * (1.0 - uf_ler) / shots)
    bf_error = np.sqrt(bf_ler * (1.0 - bf_ler) / shots)

    assert bf_failures < uf_failures
    assert bf_ler + bf_error < uf_ler - uf_error
