"""Common decoder interface used by the benchmark harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import stim


class Decoder(ABC):
    """Map detector syndromes to predicted logical-observable flips."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        **kwargs: object,
    ) -> "Decoder":
        """Construct a decoder once from a detector error model."""

    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Map ``bool[D]`` syndrome to ``bool[K]`` observable flips."""

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Map ``bool[S, D]`` syndromes to ``bool[S, K]`` predictions."""
        return np.stack([self.decode(syndrome) for syndrome in syndromes])

    def stats(self) -> dict:
        """Return optional decoder-specific metrics."""
        return {}

