"""Convert a Stim detector error model into shared decoder data structures.

Graph decoders consume the component edges, while belief propagation consumes
the whole-mechanism check matrices.  Building both views in one pass preserves
the exact correspondence needed by hybrid decoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
import stim


_MIN_PROBABILITY = 1e-15
_MAX_GRAPH_PROBABILITY = 0.5 - 1e-12


@dataclass(frozen=True)
class DecodingGraph:
    """Graph-like components of a detector error model.

    Detector nodes are ``0..D-1`` and the shared virtual boundary is node
    ``D``.  An edge's observable mask records which logical measurements are
    flipped when that component is used in a correction.
    """

    num_detectors: int
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_prob: np.ndarray
    edge_weight: np.ndarray
    edge_obs: np.ndarray
    adj_ptr: np.ndarray
    adj_edges: np.ndarray
    num_obs_conflicts: int


@dataclass(frozen=True)
class CheckMatrices:
    """Binary detector and logical effects of whole error mechanisms."""

    H: csr_matrix
    L: csr_matrix
    priors: np.ndarray
    mech_to_edges: list[np.ndarray]


@dataclass
class _MergedEdge:
    """Mutable accumulator used before arrays and CSR adjacency are frozen."""

    probability: float
    observable_mask: np.ndarray
    representative_probability: float


def _xor_target(target_set: set[int], value: int) -> None:
    """Toggle a binary target represented as a set of nonzero indices."""
    if value in target_set:
        target_set.remove(value)
    else:
        target_set.add(value)


def _split_components(
    targets: list[stim.DemTarget],
) -> list[list[stim.DemTarget]]:
    """Split one error instruction at Stim decomposition separators."""
    components: list[list[stim.DemTarget]] = [[]]
    for target in targets:
        if target.is_separator():
            components.append([])
        else:
            components[-1].append(target)
    return [component for component in components if component]


def _component_effect(
    component: list[stim.DemTarget],
) -> tuple[set[int], set[int]]:
    """Return the detector and observable effects after binary cancellation."""
    detectors: set[int] = set()
    observables: set[int] = set()
    for target in component:
        if target.is_relative_detector_id():
            _xor_target(detectors, target.val)
        elif target.is_logical_observable_id():
            _xor_target(observables, target.val)
        else:
            raise ValueError(f"unsupported target in error instruction: {target}")
    return detectors, observables


def _prefer_observable_mask(
    current: _MergedEdge,
    probability: float,
    candidate_mask: np.ndarray,
) -> None:
    """Keep the most likely component's mask, with a deterministic tie-break."""
    candidate_key = candidate_mask.tobytes()
    current_key = current.observable_mask.tobytes()
    if probability > current.representative_probability or (
        probability == current.representative_probability
        and candidate_key < current_key
    ):
        current.observable_mask = candidate_mask.copy()
        current.representative_probability = probability


def _build_adjacency(
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a CSR edge-incidence index, including the boundary node."""
    degrees = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(degrees, edge_u, 1)
    np.add.at(degrees, edge_v, 1)

    adj_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    np.cumsum(degrees, out=adj_ptr[1:])
    adj_edges = np.empty(2 * len(edge_u), dtype=np.int32)
    cursor = adj_ptr[:-1].copy()
    for edge_id, (u, v) in enumerate(zip(edge_u, edge_v)):
        adj_edges[cursor[u]] = edge_id
        cursor[u] += 1
        adj_edges[cursor[v]] = edge_id
        cursor[v] += 1
    return adj_ptr, adj_edges


def build_from_dem(
    dem: stim.DetectorErrorModel,
) -> tuple[DecodingGraph, CheckMatrices]:
    """Build graph and binary-matrix views of one decomposed Stim DEM.

    A whole ``error`` instruction is one mechanism in ``H`` and ``L``.  Each
    component separated by ``^`` is independently graph-like and becomes an
    edge.  Parallel components are combined by the probability that an odd
    number of them occurs.
    """
    flattened = dem.flattened()
    num_detectors = flattened.num_detectors
    num_observables = flattened.num_observables
    boundary = num_detectors

    h_rows: list[int] = []
    h_cols: list[int] = []
    l_rows: list[int] = []
    l_cols: list[int] = []
    priors: list[float] = []
    mechanism_edge_keys: list[list[tuple[int, int]]] = []
    merged: dict[tuple[int, int], _MergedEdge] = {}
    num_obs_conflicts = 0

    for instruction in flattened:
        if instruction.type != "error":
            continue

        mechanism_id = len(priors)
        probability = float(instruction.args_copy()[0])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid error probability {probability}")
        priors.append(probability)

        mechanism_detectors: set[int] = set()
        mechanism_observables: set[int] = set()
        edge_keys: list[tuple[int, int]] = []

        for component in _split_components(instruction.targets_copy()):
            detectors, observables = _component_effect(component)
            for detector in detectors:
                _xor_target(mechanism_detectors, detector)
            for observable in observables:
                _xor_target(mechanism_observables, observable)

            if not detectors:
                continue
            if len(detectors) > 2:
                raise ValueError(
                    "DEM is not graph-like after decomposition: "
                    f"component has detectors {sorted(detectors)}"
                )

            endpoints = sorted(detectors)
            if len(endpoints) == 1:
                endpoints.append(boundary)
            edge_key = (endpoints[0], endpoints[1])
            edge_keys.append(edge_key)

            observable_mask = np.zeros(num_observables, dtype=np.bool_)
            if observables:
                observable_mask[list(observables)] = True

            if edge_key not in merged:
                merged[edge_key] = _MergedEdge(
                    probability=probability,
                    observable_mask=observable_mask,
                    representative_probability=probability,
                )
            else:
                current = merged[edge_key]
                if not np.array_equal(current.observable_mask, observable_mask):
                    num_obs_conflicts += 1
                    _prefer_observable_mask(current, probability, observable_mask)
                current.probability = (
                    current.probability * (1.0 - probability)
                    + probability * (1.0 - current.probability)
                )

        for detector in sorted(mechanism_detectors):
            h_rows.append(detector)
            h_cols.append(mechanism_id)
        for observable in sorted(mechanism_observables):
            l_rows.append(observable)
            l_cols.append(mechanism_id)
        mechanism_edge_keys.append(edge_keys)

    edge_keys = sorted(merged)
    edge_ids = {key: edge_id for edge_id, key in enumerate(edge_keys)}
    edge_u = np.asarray([key[0] for key in edge_keys], dtype=np.int32)
    edge_v = np.asarray([key[1] for key in edge_keys], dtype=np.int32)
    edge_prob = np.asarray(
        [
            np.clip(
                merged[key].probability,
                _MIN_PROBABILITY,
                _MAX_GRAPH_PROBABILITY,
            )
            for key in edge_keys
        ],
        dtype=np.float64,
    )
    edge_weight = np.log((1.0 - edge_prob) / edge_prob)
    if edge_keys:
        edge_obs = np.stack(
            [merged[key].observable_mask for key in edge_keys]
        ).astype(np.bool_, copy=False)
    else:
        edge_obs = np.zeros((0, num_observables), dtype=np.bool_)
    adj_ptr, adj_edges = _build_adjacency(
        num_detectors + 1, edge_u, edge_v
    )

    num_mechanisms = len(priors)
    h_data = np.ones(len(h_rows), dtype=np.uint8)
    l_data = np.ones(len(l_rows), dtype=np.uint8)
    check_matrices = CheckMatrices(
        H=csr_matrix(
            (h_data, (h_rows, h_cols)),
            shape=(num_detectors, num_mechanisms),
            dtype=np.uint8,
        ),
        L=csr_matrix(
            (l_data, (l_rows, l_cols)),
            shape=(num_observables, num_mechanisms),
            dtype=np.uint8,
        ),
        priors=np.asarray(priors, dtype=np.float64),
        mech_to_edges=[
            np.asarray([edge_ids[key] for key in keys], dtype=np.int32)
            for keys in mechanism_edge_keys
        ],
    )
    graph = DecodingGraph(
        num_detectors=num_detectors,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_prob=edge_prob,
        edge_weight=edge_weight,
        edge_obs=edge_obs,
        adj_ptr=adj_ptr,
        adj_edges=adj_edges,
        num_obs_conflicts=num_obs_conflicts,
    )
    return graph, check_matrices

