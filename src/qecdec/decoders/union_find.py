"""Union-Find decoding by cluster growth followed by forest peeling.

Each detector click starts an odd cluster.  Odd clusters grow along graph
edges until they meet another odd cluster or the virtual boundary.  Peeling a
spanning forest of the grown edges then turns those neutral clusters into a
concrete correction with the requested detector syndrome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import stim

from qecdec.base import Decoder
from qecdec.dem_graph import DecodingGraph, build_from_dem


@dataclass(frozen=True)
class DecodeOutcome:
    """One decoded syndrome, including data useful for tests and profiling."""

    prediction: np.ndarray
    correction: np.ndarray
    fully_grown: np.ndarray
    growth_rounds: int
    num_clusters: int


@dataclass
class _ClusterState:
    """Mutable disjoint-set and growth state for one syndrome."""

    parent: np.ndarray
    size: np.ndarray
    parity: np.ndarray
    touches_boundary: np.ndarray
    frontier: list[set[int]]

    def find(self, node: int) -> int:
        """Return a cluster root while compressing the path to it."""
        root = node
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[node] != node:
            next_node = int(self.parent[node])
            self.parent[node] = root
            node = next_node
        return root

    def union(self, first: int, second: int) -> int:
        """Merge two clusters and combine their neutrality information."""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root

        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        self.parity[first_root] ^= self.parity[second_root]
        self.touches_boundary[first_root] |= self.touches_boundary[second_root]
        self.frontier[first_root].update(self.frontier[second_root])
        self.frontier[second_root].clear()
        return first_root

    def is_active(self, root: int) -> bool:
        """An odd cluster grows unless the boundary can absorb its defect."""
        return bool(self.parity[root] and not self.touches_boundary[root])


@dataclass
class UnionFindDecoder(Decoder):
    """Pure-Python weighted or unweighted Union-Find decoder.

    Weighted growth discretizes negative log-likelihoods so the median edge
    takes about ten support units to fill.  Unweighted growth assigns two
    units to every edge.  In either mode, two active clusters grow toward each
    other at twice the rate of one active cluster growing into neutral space.
    """

    name: ClassVar[str] = "uf"
    default_weighted: ClassVar[bool] = True

    graph: DecodingGraph
    weighted: bool = True
    weight_scale: float = field(init=False)
    growth_thresholds: np.ndarray = field(init=False, repr=False)
    _growth_rounds: list[int] = field(default_factory=list, init=False)
    _cluster_counts: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Discretize edge lengths once, not once per decoded shot."""
        if not self.weighted:
            self.weight_scale = 1.0
            self.growth_thresholds = np.full(
                len(self.graph.edge_u),
                2,
                dtype=np.int32,
            )
            return

        if len(self.graph.edge_weight) == 0:
            self.weight_scale = 1.0
            self.growth_thresholds = np.empty(0, dtype=np.int32)
            return
        median_weight = float(np.median(self.graph.edge_weight))
        self.weight_scale = 10.0 / median_weight
        self.growth_thresholds = np.maximum(
            1,
            np.rint(self.graph.edge_weight * self.weight_scale),
        ).astype(np.int32)

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        **kwargs: object,
    ) -> "UnionFindDecoder":
        """Build the shared decoding graph once from a decomposed DEM."""
        weighted = bool(kwargs.pop("weighted", cls.default_weighted))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected Union-Find options: {unexpected}")
        graph, _ = build_from_dem(dem)
        return cls(graph=graph, weighted=weighted)

    def _initial_state(self, syndrome: np.ndarray) -> _ClusterState:
        """Create singleton clusters, with detector clicks marking odd roots."""
        num_nodes = self.graph.num_detectors + 1
        boundary = self.graph.num_detectors
        parity = np.zeros(num_nodes, dtype=np.bool_)
        parity[:boundary] = syndrome
        touches_boundary = np.zeros(num_nodes, dtype=np.bool_)
        touches_boundary[boundary] = True
        frontier = [
            set(
                int(edge)
                for edge in self.graph.adj_edges[
                    self.graph.adj_ptr[node] : self.graph.adj_ptr[node + 1]
                ]
            )
            for node in range(num_nodes)
        ]
        return _ClusterState(
            parent=np.arange(num_nodes, dtype=np.int32),
            size=np.ones(num_nodes, dtype=np.int32),
            parity=parity,
            touches_boundary=touches_boundary,
            frontier=frontier,
        )

    def _validate_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """Normalize one detector sample without silently changing its shape."""
        normalized = np.asarray(syndrome, dtype=np.bool_)
        expected = (self.graph.num_detectors,)
        if normalized.shape != expected:
            raise ValueError(
                f"syndrome has shape {normalized.shape}, expected {expected}"
            )
        return normalized

    def _thresholds_for(self, weights: np.ndarray | None) -> np.ndarray:
        """Discretize optional per-shot likelihoods using the fixed graph scale.

        Belief-find changes edge likelihoods for each syndrome.  Reusing the
        scale chosen from the original DEM keeps those lengths comparable and
        makes a zero-iteration belief-find decode identical to weighted UF.
        """
        if weights is None:
            return self.growth_thresholds
        if not self.weighted:
            raise ValueError("per-shot weights require weighted Union-Find")

        normalized = np.asarray(weights, dtype=np.float64)
        expected = (len(self.graph.edge_u),)
        if normalized.shape != expected:
            raise ValueError(
                f"weights have shape {normalized.shape}, expected {expected}"
            )
        if np.any(~np.isfinite(normalized)) or np.any(normalized < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        return np.maximum(
            1,
            np.rint(normalized * self.weight_scale),
        ).astype(np.int32)

    def _grow(
        self,
        syndrome: np.ndarray,
        growth_thresholds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, _ClusterState, int]:
        """Grow odd clusters and return the edges that reached full support."""
        thresholds = (
            self.growth_thresholds
            if growth_thresholds is None
            else growth_thresholds
        )
        state = self._initial_state(syndrome)
        support = np.zeros(len(self.graph.edge_u), dtype=np.int32)
        fully_grown = np.zeros(len(self.graph.edge_u), dtype=np.bool_)
        growth_rounds = 0

        while True:
            active_roots = [
                node
                for node in range(len(state.parent))
                if state.parent[node] == node and state.is_active(node)
            ]
            if not active_roots:
                break

            edge_rates: dict[int, int] = {}
            for root in active_roots:
                for edge in tuple(state.frontier[root]):
                    first_root = state.find(int(self.graph.edge_u[edge]))
                    second_root = state.find(int(self.graph.edge_v[edge]))
                    if first_root == second_root:
                        state.frontier[root].discard(edge)
                        continue
                    if root not in (first_root, second_root):
                        state.frontier[root].discard(edge)
                        continue
                    if fully_grown[edge]:
                        continue
                    edge_rates[edge] = edge_rates.get(edge, 0) + 1

            if not edge_rates:
                raise ValueError(
                    "syndrome has an odd cluster disconnected from boundary"
                )

            rounds_to_event = min(
                (
                    int(thresholds[edge])
                    - int(support[edge])
                    + rate
                    - 1
                )
                // rate
                for edge, rate in edge_rates.items()
            )
            growth_rounds += rounds_to_event
            newly_full: list[int] = []
            for edge, rate in edge_rates.items():
                support[edge] += rounds_to_event * rate
                if support[edge] >= thresholds[edge]:
                    support[edge] = thresholds[edge]
                    fully_grown[edge] = True
                    newly_full.append(edge)

            for edge in newly_full:
                state.union(
                    int(self.graph.edge_u[edge]),
                    int(self.graph.edge_v[edge]),
                )

        return fully_grown, state, growth_rounds

    def _peel(
        self,
        syndrome: np.ndarray,
        fully_grown: np.ndarray,
    ) -> np.ndarray:
        """Peel a spanning forest to realize the syndrome as correction edges."""
        num_nodes = self.graph.num_detectors + 1
        boundary = self.graph.num_detectors
        grown_adjacency: list[list[tuple[int, int]]] = [
            [] for _ in range(num_nodes)
        ]
        for edge in np.flatnonzero(fully_grown):
            first = int(self.graph.edge_u[edge])
            second = int(self.graph.edge_v[edge])
            grown_adjacency[first].append((second, int(edge)))
            grown_adjacency[second].append((first, int(edge)))

        parent = np.full(num_nodes, -1, dtype=np.int32)
        parent_edge = np.full(num_nodes, -1, dtype=np.int32)
        order: list[int] = []
        roots = [boundary] + [node for node in range(boundary)]
        for root in roots:
            if parent[root] != -1:
                continue
            parent[root] = root
            queue = [root]
            for node in queue:
                order.append(node)
                for neighbor, edge in grown_adjacency[node]:
                    if parent[neighbor] != -1:
                        continue
                    parent[neighbor] = node
                    parent_edge[neighbor] = edge
                    queue.append(neighbor)

        residual = np.zeros(num_nodes, dtype=np.bool_)
        residual[:boundary] = syndrome
        correction = np.zeros(len(self.graph.edge_u), dtype=np.bool_)
        for node in reversed(order):
            if parent[node] == node or not residual[node]:
                continue
            edge = int(parent_edge[node])
            correction[edge] ^= True
            residual[int(parent[node])] ^= True
            residual[node] = False

        if np.any(residual[:boundary]):
            raise RuntimeError("peeling failed to reproduce the detector syndrome")
        return correction

    def decode_outcome(
        self,
        syndrome: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> DecodeOutcome:
        """Decode one sample and expose its correction and growth metrics."""
        normalized = self._validate_syndrome(syndrome)
        thresholds = self._thresholds_for(weights)
        fully_grown, state, growth_rounds = self._grow(normalized, thresholds)
        correction = self._peel(normalized, fully_grown)
        if np.any(correction):
            prediction = np.logical_xor.reduce(
                self.graph.edge_obs[correction],
                axis=0,
            )
        else:
            prediction = np.zeros(self.graph.edge_obs.shape[1], dtype=np.bool_)
        defect_nodes = np.flatnonzero(normalized)
        num_clusters = len(
            {state.find(int(node)) for node in defect_nodes}
        )
        return DecodeOutcome(
            prediction=np.asarray(prediction, dtype=np.bool_),
            correction=correction,
            fully_grown=fully_grown,
            growth_rounds=growth_rounds,
            num_clusters=num_clusters,
        )

    def decode(
        self,
        syndrome: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict logical flips by growing and peeling neutral clusters."""
        outcome = self.decode_outcome(syndrome, weights=weights)
        self._growth_rounds.append(outcome.growth_rounds)
        self._cluster_counts.append(outcome.num_clusters)
        return outcome.prediction

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch and make stats describe this batch only."""
        self._growth_rounds.clear()
        self._cluster_counts.clear()
        return super().decode_batch(syndromes)

    def stats(self) -> dict:
        """Summarize growth depth and final neutral-cluster counts."""
        if not self._growth_rounds:
            return {
                "growth_rounds": {},
                "num_clusters": {},
                "shots_decoded": 0,
            }

        growth_values, growth_counts = np.unique(
            self._growth_rounds,
            return_counts=True,
        )
        cluster_values, cluster_counts = np.unique(
            self._cluster_counts,
            return_counts=True,
        )
        return {
            "growth_rounds": {
                str(int(value)): int(count)
                for value, count in zip(growth_values, growth_counts)
            },
            "num_clusters": {
                str(int(value)): int(count)
                for value, count in zip(cluster_values, cluster_counts)
            },
            "growth_rounds_mean": float(np.mean(self._growth_rounds)),
            "growth_rounds_p50": float(np.percentile(self._growth_rounds, 50)),
            "growth_rounds_p99": float(np.percentile(self._growth_rounds, 99)),
            "num_clusters_mean": float(np.mean(self._cluster_counts)),
            "shots_decoded": len(self._growth_rounds),
            "weighted": self.weighted,
            "weight_scale": self.weight_scale,
        }


@dataclass
class UnweightedUnionFindDecoder(UnionFindDecoder):
    """Named registry variant using equal two-half-edge lengths."""

    name: ClassVar[str] = "uf_unweighted"
    default_weighted: ClassVar[bool] = False

    weighted: bool = False
