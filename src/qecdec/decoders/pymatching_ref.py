"""Untuned PyMatching reference decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim

from qecdec.base import Decoder


@dataclass
class PyMatchingDecoder(Decoder):
    """Minimum-weight perfect matching built directly from a Stim DEM."""

    name = "pymatching"

    matching: pymatching.Matching

    @classmethod
    def from_dem(
        cls,
        dem: stim.DetectorErrorModel,
        **kwargs: object,
    ) -> "PyMatchingDecoder":
        """Build PyMatching's standard DEM-derived matching graph once."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected PyMatching options: {unexpected}")
        return cls(pymatching.Matching.from_detector_error_model(dem))

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Predict the observable flips for one detector syndrome."""
        return np.asarray(self.matching.decode(syndrome), dtype=np.bool_)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Use PyMatching's native batch decoder for all supplied shots."""
        predictions = self.matching.decode_batch(syndromes)
        return np.asarray(predictions, dtype=np.bool_)

