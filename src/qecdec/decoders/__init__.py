"""Decoder implementations available to the common harness."""

from qecdec.base import Decoder
from qecdec.decoders.pymatching_ref import PyMatchingDecoder


REGISTRY: dict[str, type[Decoder]] = {
    PyMatchingDecoder.name: PyMatchingDecoder,
}

__all__ = ["PyMatchingDecoder", "REGISTRY"]

