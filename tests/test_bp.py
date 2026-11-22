"""Correctness and batching checks for normalized min-sum BP."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from qecdec.circuits import make_circuit
from qecdec.decoders.bp import BPDecoder
from qecdec.dem_graph import CheckMatrices, build_from_dem


def _matrices(
    h: list[list[int]],
    l: list[list[int]],
    priors: list[float],
) -> CheckMatrices:
    """Build minimal whole-mechanism matrices for a hand-worked code."""
    h_array = np.asarray(h, dtype=np.uint8)
    l_array = np.asarray(l, dtype=np.uint8)
    return CheckMatrices(
        H=csr_matrix(h_array),
        L=csr_matrix(l_array),
        priors=np.asarray(priors, dtype=np.float64),
        mech_to_edges=[np.empty(0, dtype=np.int32) for _ in priors],
    )


def _repetition_matrices() -> CheckMatrices:
    """Return a five-edge repetition chain whose Tanner graph is a tree."""
    return _matrices(
        h=[
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        l=[[0, 0, 0, 0, 1]],
        priors=[0.05] * 5,
    )


def test_zero_syndrome_converges_without_messages() -> None:
    decoder = BPDecoder(_repetition_matrices())

    prediction = decoder.decode(np.zeros(4, dtype=np.bool_))

    np.testing.assert_array_equal(prediction, [False])
    np.testing.assert_array_equal(decoder.last_converged, [True])
    np.testing.assert_array_equal(decoder.last_iters, [0])


def test_tree_repetition_code_decodes_every_single_error() -> None:
    matrices = _repetition_matrices()
    decoder = BPDecoder(matrices, alpha=1.0)
    syndromes = matrices.H.T.toarray().astype(np.bool_)
    expected = matrices.L.T.toarray().astype(np.bool_)

    predictions = decoder.decode_batch(syndromes)

    np.testing.assert_array_equal(predictions, expected)
    assert np.all(decoder.last_converged)


def test_one_iteration_check_messages_match_hand_calculation() -> None:
    matrices = _matrices(
        h=[[1, 1, 1], [0, 1, 1]],
        l=[[0, 0, 0]],
        priors=[0.1, 0.1, 0.1],
    )
    decoder = BPDecoder(matrices, alpha=1.0)
    variable_to_check = np.asarray(
        [[2.0, -3.0, 4.0, 5.0, -7.0]],
        dtype=np.float64,
    )
    syndrome = np.asarray([[False, True]], dtype=np.bool_)

    actual = decoder._check_update(variable_to_check, syndrome)

    np.testing.assert_array_equal(actual, [[-3.0, 2.0, -2.0, 7.0, -5.0]])


def test_batched_decode_matches_looped_shots_bit_for_bit() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    _, matrices = build_from_dem(dem)
    syndromes = matrices.H.T.toarray().astype(np.bool_)[:32]
    batched = BPDecoder.from_dem(dem, batch_size=7)
    looped = BPDecoder.from_dem(dem)

    batch_predictions = batched.decode_batch(syndromes)
    batch_converged = batched.last_converged.copy()
    batch_iters = batched.last_iters.copy()
    batch_posterior = batched.last_posterior_llr.copy()
    loop_predictions = []
    loop_converged = []
    loop_iters = []
    loop_posterior = []
    for syndrome in syndromes:
        loop_predictions.append(looped.decode(syndrome))
        loop_converged.append(looped.last_converged[0])
        loop_iters.append(looped.last_iters[0])
        loop_posterior.append(looped.last_posterior_llr[0].copy())

    np.testing.assert_array_equal(batch_predictions, loop_predictions)
    np.testing.assert_array_equal(batch_converged, loop_converged)
    np.testing.assert_array_equal(batch_iters, loop_iters)
    np.testing.assert_array_equal(batch_posterior, loop_posterior)


def test_single_fault_rate_and_soft_outputs_are_reportable() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    _, matrices = build_from_dem(dem)
    decoder = BPDecoder.from_dem(dem)
    syndromes = matrices.H.T.toarray().astype(np.bool_)

    predictions = decoder.decode_batch(syndromes)
    stats = decoder.stats()

    assert predictions.shape == matrices.L.T.shape
    assert decoder.last_converged.shape == (matrices.H.shape[1],)
    assert decoder.last_iters.shape == (matrices.H.shape[1],)
    assert decoder.last_posterior_llr.shape == (
        matrices.H.shape[1],
        matrices.H.shape[1],
    )
    assert 0.0 <= stats["convergence_rate"] <= 1.0
    assert 0.0 <= stats["mean_iterations"] <= decoder.max_iter
