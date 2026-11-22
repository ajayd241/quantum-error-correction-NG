"""Belief-find decoding, introduced by Higgott et al. (PRX 13, 031007).

Belief propagation first turns one detector syndrome into posterior fault
likelihoods.  Those beliefs reweight the component graph before weighted
Union-Find grows and peels its correction.  The optional BP-converged fast
path treats BP as a partial decoder, following Caune et al. (arXiv:2306.17142).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from scipy.sparse import csr_matrix
import stim

from qecdec.base import Decoder
from qecdec.decoders.bp import BPDecoder
from qecdec.decoders.union_find import UnionFindDecoder
from qecdec.dem_graph import CheckMatrices, DecodingGraph, build_from_dem


_MIN_PROBABILITY = 1e-15
_MAX_UF_PROBABILITY = 0.5 - 1e-12
_MIN_POSTERIOR_PROBABILITY = 1e-12


@dataclass
class BeliefFindDecoder(Decoder):
    """Use BP's syndrome-specific beliefs to guide weighted Union-Find."""

    name: ClassVar[str] = "belief_find"

    graph: DecodingGraph
    matrices: CheckMatrices
    bp_iters: int = 30
    fast_path: bool = True
    alpha: float = 0.625

    edge_to_mechanism: csr_matrix = field(init=False, repr=False)
    bp: BPDecoder = field(init=False, repr=False)
    union_find: UnionFindDecoder = field(init=False, repr=False)
    last_fast_path: np.ndarray = field(init=False, repr=False)
    last_bp_iters: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the mechanism-to-edge incidence map and both component decoders."""
        num_edges = len(self.graph.edge_u)
        num_mechanisms = int(self.matrices.H.shape[1])
        if len(self.matrices.mech_to_edges) != num_mechanisms:
            raise ValueError(
                "mech_to_edges has length "
                f"{len(self.matrices.mech_to_edges)}, expected {num_mechanisms}"
            )

        rows: list[int] = []
        cols: list[int] = []
        for mechanism, edge_ids in enumerate(self.matrices.mech_to_edges):
            unique_edges = np.unique(np.asarray(edge_ids, dtype=np.int32))
            if np.any(unique_edges < 0) or np.any(unique_edges >= num_edges):
                raise ValueError(f"mechanism {mechanism} has an invalid edge id")
            rows.extend(int(edge) for edge in unique_edges)
            cols.extend([mechanism] * len(unique_edges))
        self.edge_to_mechanism = csr_matrix(
            (
                np.ones(len(rows), dtype=np.uint8),
                (rows, cols),
            ),
            shape=(num_edges, num_mechanisms),
            dtype=np.uint8,
        )
        if num_edges and np.any(self.edge_to_mechanism.getnnz(axis=1) == 0):
            raise ValueError("every graph edge must belong to a DEM mechanism")

        self.bp = BPDecoder(
            matrices=self.matrices,
            max_iter=self.bp_iters,
            alpha=self.alpha,
        )
        self.union_find = UnionFindDecoder(graph=self.graph, weighted=True)
        self.last_fast_path = np.empty(0, dtype=np.bool_)
        self.last_bp_iters = np.empty(0, dtype=np.int32)

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        **kwargs: object,
    ) -> "BeliefFindDecoder":
        """Build both correlated-mechanism and component-edge DEM views once."""
        graph, matrices = build_from_dem(dem)
        return cls(graph=graph, matrices=matrices, **kwargs)

    def edge_probabilities(self, mechanism_probabilities: np.ndarray) -> np.ndarray:
        """XOR-combine posterior mechanism probabilities onto graph edges.

        For independent Bernoulli mechanisms, ``1 - 2p`` multiplies under XOR.
        Logs preserve numerical precision when many mechanisms share an edge;
        a separate sparse parity product retains the sign when a posterior is
        greater than one half.
        """
        probabilities = np.asarray(mechanism_probabilities, dtype=np.float64)
        expected_width = int(self.matrices.H.shape[1])
        if probabilities.ndim not in (1, 2) or probabilities.shape[-1] != expected_width:
            raise ValueError(
                "mechanism probabilities have shape "
                f"{probabilities.shape}, expected ({expected_width},) or "
                f"(shots, {expected_width})"
            )
        if np.any(~np.isfinite(probabilities)):
            raise ValueError("mechanism probabilities must be finite")

        was_vector = probabilities.ndim == 1
        batch = probabilities[np.newaxis, :] if was_vector else probabilities
        clipped = np.clip(
            batch,
            _MIN_POSTERIOR_PROBABILITY,
            1.0 - _MIN_POSTERIOR_PROBABILITY,
        )
        centered = 1.0 - 2.0 * clipped
        log_magnitude = self.edge_to_mechanism @ np.log(
            np.maximum(np.abs(centered), np.finfo(np.float64).tiny)
        ).T
        negative_count = self.edge_to_mechanism @ (centered < 0.0).astype(
            np.int32
        ).T
        sign = np.where(np.asarray(negative_count) % 2, -1.0, 1.0)
        product = sign * np.exp(np.asarray(log_magnitude))
        edge_probabilities = (0.5 * (1.0 - product)).T
        return edge_probabilities[0] if was_vector else edge_probabilities

    @staticmethod
    def _weights_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
        """Convert posteriors into nonnegative UF lengths."""
        clipped = np.clip(
            probabilities,
            _MIN_PROBABILITY,
            _MAX_UF_PROBABILITY,
        )
        return np.log((1.0 - clipped) / clipped)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Run batched BP, then UF only for shots that miss the fast path."""
        normalized = self.bp._validate_syndromes(syndromes)
        bp_predictions = self.bp.decode_batch(normalized)
        num_shots = len(normalized)
        self.last_bp_iters = self.bp.last_iters.copy()
        self.last_fast_path = (
            self.bp.last_converged.copy()
            if self.fast_path
            else np.zeros(num_shots, dtype=np.bool_)
        )

        predictions = np.empty_like(bp_predictions)
        predictions[self.last_fast_path] = bp_predictions[self.last_fast_path]
        uf_indices = np.flatnonzero(~self.last_fast_path)
        if len(uf_indices) == 0:
            return predictions

        posterior = self.bp.last_posterior_llr[uf_indices]
        mechanism_probabilities = 1.0 / (1.0 + np.exp(posterior))
        edge_probabilities = self.edge_probabilities(mechanism_probabilities)
        edge_weights = self._weights_from_probabilities(edge_probabilities)
        for local_index, shot_index in enumerate(uf_indices):
            predictions[shot_index] = self.union_find.decode(
                normalized[shot_index],
                weights=edge_weights[local_index],
            )
        return predictions

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode one detector sample while retaining one-shot hybrid stats."""
        normalized = np.asarray(syndrome, dtype=np.bool_)
        expected = (self.graph.num_detectors,)
        if normalized.shape != expected:
            raise ValueError(
                f"syndrome has shape {normalized.shape}, expected {expected}"
            )
        return self.decode_batch(normalized[np.newaxis, :])[0]

    def stats(self) -> dict:
        """Report how often BP finished early and its iteration cost."""
        if len(self.last_bp_iters) == 0:
            return {
                "fast_path_fraction": 0.0,
                "mean_bp_iterations": 0.0,
                "shots_decoded": 0,
                "bp_iters": self.bp_iters,
                "fast_path": self.fast_path,
                "alpha": self.alpha,
            }
        return {
            "fast_path_fraction": float(np.mean(self.last_fast_path)),
            "mean_bp_iterations": float(np.mean(self.last_bp_iters)),
            "shots_decoded": len(self.last_bp_iters),
            "bp_iters": self.bp_iters,
            "fast_path": self.fast_path,
            "alpha": self.alpha,
        }
