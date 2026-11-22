"""Batched normalized min-sum belief propagation on DEM mechanisms.

Each variable is one complete detector-error-model mechanism, including any
correlated components separated by ``^``.  Check nodes exchange likelihood
messages with those mechanisms until the hard decision explains the observed
detector syndrome or the iteration budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import stim

from qecdec.base import Decoder
from qecdec.dem_graph import CheckMatrices, build_from_dem


@dataclass
class BPDecoder(Decoder):
    """Normalized min-sum BP, vectorized over Tanner edges and shots."""

    name: ClassVar[str] = "bp"

    matrices: CheckMatrices
    max_iter: int = 30
    alpha: float = 0.75
    batch_size: int = 256
    llr_clip: float = 50.0

    chk: np.ndarray = field(init=False, repr=False)
    var: np.ndarray = field(init=False, repr=False)
    check_ptr: np.ndarray = field(init=False, repr=False)
    var_order: np.ndarray = field(init=False, repr=False)
    var_ptr: np.ndarray = field(init=False, repr=False)
    prior_llr: np.ndarray = field(init=False, repr=False)
    last_converged: np.ndarray = field(init=False, repr=False)
    last_iters: np.ndarray = field(init=False, repr=False)
    last_posterior_llr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the two Tanner-edge orderings used by message reductions."""
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be nonnegative, got {self.max_iter}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if not np.isfinite(self.llr_clip) or self.llr_clip <= 0.0:
            raise ValueError(
                f"llr_clip must be finite and positive, got {self.llr_clip}"
            )

        matrix = self.matrices.H.tocsr()
        if np.any(matrix.data != 1):
            raise ValueError("H must be a binary matrix with explicit ones")
        num_checks, num_variables = matrix.shape
        self.chk = np.repeat(
            np.arange(num_checks, dtype=np.int32),
            np.diff(matrix.indptr),
        )
        self.var = matrix.indices.astype(np.int32, copy=True)
        self.check_ptr = matrix.indptr.astype(np.int32, copy=True)
        self.var_order = np.argsort(self.var, kind="stable").astype(
            np.int32,
            copy=False,
        )
        variable_degrees = np.bincount(
            self.var,
            minlength=num_variables,
        ).astype(np.int32, copy=False)
        self.var_ptr = np.zeros(num_variables + 1, dtype=np.int32)
        np.cumsum(variable_degrees, out=self.var_ptr[1:])

        probabilities = np.clip(
            np.asarray(self.matrices.priors, dtype=np.float64),
            1e-15,
            1.0 - 1e-15,
        )
        if probabilities.shape != (num_variables,):
            raise ValueError(
                "priors have shape "
                f"{probabilities.shape}, expected {(num_variables,)}"
            )
        self.prior_llr = np.clip(
            np.log((1.0 - probabilities) / probabilities),
            -self.llr_clip,
            self.llr_clip,
        )
        self.last_converged = np.empty(0, dtype=np.bool_)
        self.last_iters = np.empty(0, dtype=np.int32)
        self.last_posterior_llr = np.empty(
            (0, num_variables),
            dtype=np.float64,
        )

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        **kwargs: object,
    ) -> "BPDecoder":
        """Build BP from whole error mechanisms in a decomposed DEM."""
        _, matrices = build_from_dem(dem)
        return cls(matrices=matrices, **kwargs)

    @property
    def num_checks(self) -> int:
        """Number of detector constraints in the Tanner graph."""
        return int(self.matrices.H.shape[0])

    @property
    def num_variables(self) -> int:
        """Number of whole DEM error mechanisms considered by BP."""
        return int(self.matrices.H.shape[1])

    def _validate_syndromes(self, syndromes: np.ndarray) -> np.ndarray:
        """Normalize a detector batch without silently changing its shape."""
        normalized = np.asarray(syndromes, dtype=np.bool_)
        expected_width = self.num_checks
        if normalized.ndim != 2 or normalized.shape[1] != expected_width:
            raise ValueError(
                "syndromes have shape "
                f"{normalized.shape}, expected (shots, {expected_width})"
            )
        return normalized

    def _check_update(
        self,
        variable_to_check: np.ndarray,
        syndromes: np.ndarray,
    ) -> np.ndarray:
        """Send each check's parity evidence back along every Tanner edge."""
        num_shots, num_edges = variable_to_check.shape
        if num_edges != len(self.chk):
            raise ValueError(
                "variable-to-check messages have width "
                f"{num_edges}, expected {len(self.chk)}"
            )
        if num_edges == 0:
            return np.empty((num_shots, 0), dtype=np.float64)

        check_degrees = np.diff(self.check_ptr)
        nonempty_checks = np.flatnonzero(check_degrees)
        starts = self.check_ptr[nonempty_checks]
        absolute = np.abs(variable_to_check)

        min1_present = np.minimum.reduceat(absolute, starts, axis=1)
        min1 = np.full(
            (num_shots, self.num_checks),
            self.llr_clip,
            dtype=np.float64,
        )
        min1[:, nonempty_checks] = min1_present
        min1_by_edge = min1[:, self.chk]

        is_minimum = absolute == min1_by_edge
        edge_indices = np.broadcast_to(
            np.arange(num_edges, dtype=np.int32),
            (num_shots, num_edges),
        )
        argmin_present = np.minimum.reduceat(
            np.where(is_minimum, edge_indices, num_edges),
            starts,
            axis=1,
        )
        argmin = np.full(
            (num_shots, self.num_checks),
            num_edges,
            dtype=np.int32,
        )
        argmin[:, nonempty_checks] = argmin_present
        use_second = edge_indices == argmin[:, self.chk]

        without_argmin = np.where(use_second, np.inf, absolute)
        min2_present = np.minimum.reduceat(
            without_argmin,
            starts,
            axis=1,
        )
        min2_present = np.where(
            np.isfinite(min2_present),
            min2_present,
            self.llr_clip,
        )
        min2 = np.full_like(min1, self.llr_clip)
        min2[:, nonempty_checks] = min2_present
        magnitudes = np.where(
            use_second,
            min2[:, self.chk],
            min1_by_edge,
        )

        negative = variable_to_check < 0.0
        negative_count_present = np.add.reduceat(
            negative,
            starts,
            axis=1,
        )
        negative_parity = np.zeros(
            (num_shots, self.num_checks),
            dtype=np.bool_,
        )
        negative_parity[:, nonempty_checks] = (
            negative_count_present % 2
        ).astype(np.bool_)
        product_sign = np.where(negative_parity[:, self.chk], -1.0, 1.0)
        own_sign = np.where(negative, -1.0, 1.0)
        syndrome_sign = np.where(syndromes[:, self.chk], -1.0, 1.0)
        return np.clip(
            self.alpha
            * syndrome_sign
            * product_sign
            * own_sign
            * magnitudes,
            -self.llr_clip,
            self.llr_clip,
        )

    def _sum_by_variable(self, messages: np.ndarray) -> np.ndarray:
        """Sum check messages incident on each error-mechanism variable."""
        num_shots = messages.shape[0]
        totals = np.zeros(
            (num_shots, self.num_variables),
            dtype=np.float64,
        )
        if messages.shape[1] == 0:
            return totals
        variable_degrees = np.diff(self.var_ptr)
        nonempty_variables = np.flatnonzero(variable_degrees)
        starts = self.var_ptr[nonempty_variables]
        totals[:, nonempty_variables] = np.add.reduceat(
            messages[:, self.var_order],
            starts,
            axis=1,
        )
        return totals

    def _satisfies(
        self,
        hard_decision: np.ndarray,
        syndromes: np.ndarray,
    ) -> np.ndarray:
        """Return which proposed fault sets reproduce their detector data."""
        produced = np.asarray(
            self.matrices.H @ hard_decision.astype(np.int32).T
        ).T
        return np.all((produced % 2).astype(np.bool_) == syndromes, axis=1)

    def _predict(self, hard_decision: np.ndarray) -> np.ndarray:
        """Map hard mechanism decisions to logical-observable flips."""
        logical_counts = np.asarray(
            self.matrices.L @ hard_decision.astype(np.int32).T
        ).T
        return (logical_counts % 2).astype(np.bool_)

    def _decode_chunk(
        self,
        syndromes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode one memory-bounded chunk while vectorizing all its shots."""
        num_shots = len(syndromes)
        posterior = np.broadcast_to(
            self.prior_llr,
            (num_shots, self.num_variables),
        ).copy()
        hard_decision = posterior < 0.0
        converged = self._satisfies(hard_decision, syndromes)
        iterations = np.full(num_shots, self.max_iter, dtype=np.int32)
        iterations[converged] = 0
        variable_to_check = np.broadcast_to(
            self.prior_llr[self.var],
            (num_shots, len(self.var)),
        ).copy()

        for iteration in range(1, self.max_iter + 1):
            active_indices = np.flatnonzero(~converged)
            if len(active_indices) == 0:
                break
            check_to_variable = self._check_update(
                variable_to_check[active_indices],
                syndromes[active_indices],
            )
            variable_totals = self._sum_by_variable(check_to_variable)
            updated_posterior = np.clip(
                self.prior_llr + variable_totals,
                -self.llr_clip,
                self.llr_clip,
            )
            posterior[active_indices] = updated_posterior
            hard_decision[active_indices] = updated_posterior < 0.0

            active_satisfied = self._satisfies(
                hard_decision[active_indices],
                syndromes[active_indices],
            )
            newly_converged = active_indices[active_satisfied]
            iterations[newly_converged] = iteration
            converged[newly_converged] = True

            still_active = active_indices[~active_satisfied]
            if len(still_active) > 0:
                updated_messages = np.clip(
                    self.prior_llr[self.var]
                    + variable_totals[:, self.var]
                    - check_to_variable,
                    -self.llr_clip,
                    self.llr_clip,
                )
                variable_to_check[still_active] = updated_messages[
                    ~active_satisfied
                ]

        return (
            self._predict(hard_decision),
            converged,
            iterations,
            posterior,
        )

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode shots in deterministic chunks and retain their soft beliefs."""
        normalized = self._validate_syndromes(syndromes)
        num_shots = len(normalized)
        num_observables = int(self.matrices.L.shape[0])
        predictions = np.empty(
            (num_shots, num_observables),
            dtype=np.bool_,
        )
        self.last_converged = np.empty(num_shots, dtype=np.bool_)
        self.last_iters = np.empty(num_shots, dtype=np.int32)
        self.last_posterior_llr = np.empty(
            (num_shots, self.num_variables),
            dtype=np.float64,
        )
        for start in range(0, num_shots, self.batch_size):
            stop = min(start + self.batch_size, num_shots)
            (
                predictions[start:stop],
                self.last_converged[start:stop],
                self.last_iters[start:stop],
                self.last_posterior_llr[start:stop],
            ) = self._decode_chunk(normalized[start:stop])
        return predictions

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode one syndrome through the same path as a one-shot batch."""
        normalized = np.asarray(syndrome, dtype=np.bool_)
        if normalized.shape != (self.num_checks,):
            raise ValueError(
                f"syndrome has shape {normalized.shape}, "
                f"expected {(self.num_checks,)}"
            )
        return self.decode_batch(normalized[np.newaxis, :])[0]

    def stats(self) -> dict:
        """Report how often and how quickly the hard decisions became valid."""
        if len(self.last_converged) == 0:
            return {
                "convergence_rate": 0.0,
                "mean_iterations": 0.0,
                "shots_decoded": 0,
                "max_iter": self.max_iter,
                "alpha": self.alpha,
            }
        return {
            "convergence_rate": float(np.mean(self.last_converged)),
            "mean_iterations": float(np.mean(self.last_iters)),
            "converged_mean_iterations": (
                float(np.mean(self.last_iters[self.last_converged]))
                if np.any(self.last_converged)
                else None
            ),
            "shots_decoded": len(self.last_converged),
            "max_iter": self.max_iter,
            "alpha": self.alpha,
        }
