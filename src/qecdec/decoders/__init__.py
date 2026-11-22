"""Decoder implementations available to the common harness."""

from qecdec.base import Decoder
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.decoders.union_find import (
    UnionFindDecoder,
    UnweightedUnionFindDecoder,
)


REGISTRY: dict[str, type[Decoder]] = {
    PyMatchingDecoder.name: PyMatchingDecoder,
    UnionFindDecoder.name: UnionFindDecoder,
    UnweightedUnionFindDecoder.name: UnweightedUnionFindDecoder,
}

__all__ = [
    "PyMatchingDecoder",
    "UnionFindDecoder",
    "UnweightedUnionFindDecoder",
    "REGISTRY",
]

