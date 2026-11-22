"""Correctness checks for Union-Find cluster growth and peeling."""

from __future__ import annotations

import numpy as np
import pytest

from qecdec.circuits import make_circuit
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.decoders.union_find import (
    UnionFindDecoder,
    UnweightedUnionFindDecoder,
)
from qecdec.dem_graph import DecodingGraph, build_from_dem
from qecdec.harness import sample


def _toy_repetition_graph() -> DecodingGraph:
    """Return a five-edge repetition-code chain with a shared boundary."""
    num_detectors = 4
    edge_u = np.asarray([0, 0, 1, 2, 3], dtype=np.int32)
    edge_v = np.asarray([1, 4, 2, 3, 4], dtype=np.int32)
    edge_prob = np.full(5, 0.05, dtype=np.float64)
    edge_weight = np.log((1.0 - edge_prob) / edge_prob)
    edge_obs = np.asarray(
        [[False], [False], [False], [False], [True]],
        dtype=np.bool_,
    )

    incident: list[list[int]] = [[] for _ in range(num_detectors + 1)]
    for edge, (first, second) in enumerate(zip(edge_u, edge_v)):
        incident[int(first)].append(edge)
        incident[int(second)].append(edge)
    adj_ptr = np.zeros(num_detectors + 2, dtype=np.int32)
    for node, edges in enumerate(incident):
        adj_ptr[node + 1] = adj_ptr[node] + len(edges)
    adj_edges = np.asarray(
        [edge for edges in incident for edge in edges],
        dtype=np.int32,
    )
    return DecodingGraph(
        num_detectors=num_detectors,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_prob=edge_prob,
        edge_weight=edge_weight,
        edge_obs=edge_obs,
        adj_ptr=adj_ptr,
        adj_edges=adj_edges,
        num_obs_conflicts=0,
    )


def _syndrome_of_edges(
    graph: DecodingGraph,
    correction: np.ndarray,
) -> np.ndarray:
    """Compute detector parity produced by a graph-edge selection."""
    syndrome = np.zeros(graph.num_detectors, dtype=np.bool_)
    for edge in np.flatnonzero(correction):
        first = int(graph.edge_u[edge])
        second = int(graph.edge_v[edge])
        syndrome[first] ^= True
        if second < graph.num_detectors:
            syndrome[second] ^= True
    return syndrome


@pytest.mark.parametrize(
    "decoder_type",
    [UnionFindDecoder, UnweightedUnionFindDecoder],
    ids=["weighted", "unweighted"],
)
def test_zero_syndrome_has_zero_prediction_and_growth(
    decoder_type: type[UnionFindDecoder],
) -> None:
    graph = _toy_repetition_graph()
    decoder = decoder_type(graph)

    outcome = decoder.decode_outcome(
        np.zeros(graph.num_detectors, dtype=np.bool_)
    )

    np.testing.assert_array_equal(outcome.prediction, [False])
    assert not np.any(outcome.correction)
    assert outcome.growth_rounds == 0
    assert outcome.num_clusters == 0


@pytest.mark.parametrize(
    ("physical_edges", "expected_observable"),
    [
        ([0], False),
        ([2], False),
        ([1], False),
        ([4], True),
        ([0, 2], False),
    ],
)
@pytest.mark.parametrize(
    "decoder_type",
    [UnionFindDecoder, UnweightedUnionFindDecoder],
    ids=["weighted", "unweighted"],
)
def test_repetition_code_errors_are_hand_decoded(
    physical_edges: list[int],
    expected_observable: bool,
    decoder_type: type[UnionFindDecoder],
) -> None:
    graph = _toy_repetition_graph()
    decoder = decoder_type(graph)
    physical = np.zeros(len(graph.edge_u), dtype=np.bool_)
    physical[physical_edges] = True
    syndrome = _syndrome_of_edges(graph, physical)

    outcome = decoder.decode_outcome(syndrome)

    np.testing.assert_array_equal(
        outcome.prediction,
        [expected_observable],
    )
    np.testing.assert_array_equal(
        _syndrome_of_edges(graph, outcome.correction),
        syndrome,
    )


@pytest.mark.parametrize(
    "decoder_type",
    [UnionFindDecoder, UnweightedUnionFindDecoder],
    ids=["weighted", "unweighted"],
)
@pytest.mark.parametrize("distance", [3, 5])
def test_union_find_corrects_every_single_fault(
    distance: int,
    decoder_type: type[UnionFindDecoder],
) -> None:
    circuit = make_circuit(distance, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    _, matrices = build_from_dem(dem)
    decoder = decoder_type.from_dem(dem)
    syndromes = matrices.H.T.toarray().astype(np.bool_)
    expected = matrices.L.T.toarray().astype(np.bool_)

    predictions = decoder.decode_batch(syndromes)

    np.testing.assert_array_equal(predictions, expected)


@pytest.mark.parametrize(
    "decoder_type",
    [UnionFindDecoder, UnweightedUnionFindDecoder],
    ids=["weighted", "unweighted"],
)
def test_peeling_reproduces_random_sampled_syndromes(
    decoder_type: type[UnionFindDecoder],
) -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)
    decoder = decoder_type.from_dem(dem)
    syndromes, _ = sample(circuit, shots=1_000, seed=17)

    for syndrome in syndromes:
        outcome = decoder.decode_outcome(syndrome)
        np.testing.assert_array_equal(
            _syndrome_of_edges(decoder.graph, outcome.correction),
            syndrome,
        )


def test_weighted_growth_discretizes_median_edge_to_about_ten() -> None:
    circuit = make_circuit(3, 0.005)
    dem = circuit.detector_error_model(decompose_errors=True)

    weighted = UnionFindDecoder.from_dem(dem)
    unweighted = UnweightedUnionFindDecoder.from_dem(dem)

    assert weighted.weighted
    assert 9 <= np.median(weighted.growth_thresholds) <= 11
    assert not unweighted.weighted
    np.testing.assert_array_equal(unweighted.growth_thresholds, 2)


def test_weighted_union_find_agrees_with_pymatching_at_low_noise() -> None:
    circuit = make_circuit(5, 0.001)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes, _ = sample(circuit, shots=10_000, seed=123)
    union_find = UnionFindDecoder.from_dem(dem)
    pymatching = PyMatchingDecoder.from_dem(dem)

    uf_predictions = union_find.decode_batch(syndromes)
    pm_predictions = pymatching.decode_batch(syndromes)

    agreement = np.mean(np.all(uf_predictions == pm_predictions, axis=1))
    assert agreement >= 0.99


@pytest.mark.slow
@pytest.mark.parametrize("distance", [3, 5])
def test_weighted_union_find_accuracy_on_circuit_noise(distance: int) -> None:
    circuit = make_circuit(distance, 0.003)
    dem = circuit.detector_error_model(decompose_errors=True)
    syndromes, observables = sample(
        circuit,
        shots=20_000,
        seed=3_100 + distance,
    )
    decoders = [
        PyMatchingDecoder.from_dem(dem),
        UnionFindDecoder.from_dem(dem),
        UnweightedUnionFindDecoder.from_dem(dem),
    ]
    failures = {}
    for decoder in decoders:
        predictions = decoder.decode_batch(syndromes)
        failures[decoder.name] = int(
            np.count_nonzero(np.any(predictions != observables, axis=1))
        )

    assert failures["uf"] < failures["uf_unweighted"]
    assert failures["uf"] <= 3 * failures["pymatching"]
