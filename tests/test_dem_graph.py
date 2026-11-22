"""Structural and single-fault checks for the shared DEM representation."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qecdec.circuits import make_circuit
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.dem_graph import build_from_dem

PHYSICAL_ERROR_RATE = 0.005


def _make_dem(distance: int) -> stim.DetectorErrorModel:
    circuit = make_circuit(distance, PHYSICAL_ERROR_RATE)
    return circuit.detector_error_model(decompose_errors=True)


def _toggle(values: set[int], value: int) -> None:
    if value in values:
        values.remove(value)
    else:
        values.add(value)


def _raw_mechanism_effects(
    dem: stim.DetectorErrorModel,
) -> tuple[list[set[int]], list[set[int]]]:
    """Read whole-instruction effects without using qecdec's parser."""
    detector_effects: list[set[int]] = []
    observable_effects: list[set[int]] = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        detectors: set[int] = set()
        observables: set[int] = set()
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                _toggle(detectors, target.val)
            elif target.is_logical_observable_id():
                _toggle(observables, target.val)
        detector_effects.append(detectors)
        observable_effects.append(observables)
    return detector_effects, observable_effects


@pytest.mark.parametrize("distance", [3, 5])
def test_graph_structure_and_probabilities(distance: int) -> None:
    dem = _make_dem(distance)
    graph, matrices = build_from_dem(dem)
    num_edges = len(graph.edge_u)

    assert graph.num_detectors == dem.num_detectors
    assert graph.adj_ptr.shape == (dem.num_detectors + 2,)
    assert graph.adj_edges.shape == (2 * num_edges,)
    assert np.all((0 <= graph.edge_u) & (graph.edge_u < dem.num_detectors))
    assert np.all((graph.edge_u < graph.edge_v) & (graph.edge_v <= dem.num_detectors))
    assert np.all((0.0 < graph.edge_prob) & (graph.edge_prob < 0.5))
    np.testing.assert_allclose(
        graph.edge_weight,
        np.log((1.0 - graph.edge_prob) / graph.edge_prob),
    )
    assert graph.edge_obs.shape == (num_edges, dem.num_observables)
    assert matrices.H.shape[0] == dem.num_detectors
    assert matrices.L.shape[0] == dem.num_observables

    incidence_counts = np.bincount(graph.adj_edges, minlength=num_edges)
    np.testing.assert_array_equal(incidence_counts, np.full(num_edges, 2))


@pytest.mark.parametrize("distance", [3, 5])
def test_check_columns_match_raw_error_targets(distance: int) -> None:
    dem = _make_dem(distance)
    _, matrices = build_from_dem(dem)
    detector_effects, observable_effects = _raw_mechanism_effects(dem)

    assert len(detector_effects) == matrices.H.shape[1]
    assert len(observable_effects) == matrices.L.shape[1]
    for mechanism, expected in enumerate(detector_effects):
        actual = set(matrices.H.getcol(mechanism).nonzero()[0])
        assert actual == expected
    for mechanism, expected in enumerate(observable_effects):
        actual = set(matrices.L.getcol(mechanism).nonzero()[0])
        assert actual == expected


@pytest.mark.parametrize("distance", [3, 5])
def test_mechanism_mapping_covers_every_graph_edge(distance: int) -> None:
    dem = _make_dem(distance)
    graph, matrices = build_from_dem(dem)

    assert len(matrices.mech_to_edges) == matrices.H.shape[1]
    covered = {
        int(edge)
        for mechanism_edges in matrices.mech_to_edges
        for edge in mechanism_edges
    }
    assert covered == set(range(len(graph.edge_u)))


@pytest.mark.parametrize("distance", [3, 5])
def test_pymatching_corrects_every_single_fault(distance: int) -> None:
    dem = _make_dem(distance)
    _, matrices = build_from_dem(dem)
    decoder = PyMatchingDecoder.from_dem(dem)

    syndromes = matrices.H.T.toarray().astype(np.bool_)
    expected = matrices.L.T.toarray().astype(np.bool_)
    predictions = decoder.decode_batch(syndromes)

    np.testing.assert_array_equal(predictions, expected)

