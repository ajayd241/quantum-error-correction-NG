"""Decoder implementations available to the common harness."""

from qecdec.base import Decoder
from qecdec.decoders.bp import BPDecoder
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.decoders.union_find import (
    UnionFindDecoder,
    UnweightedUnionFindDecoder,
)


REGISTRY: dict[str, type[Decoder]] = {
    BPDecoder.name: BPDecoder,
    PyMatchingDecoder.name: PyMatchingDecoder,
    UnionFindDecoder.name: UnionFindDecoder,
    UnweightedUnionFindDecoder.name: UnweightedUnionFindDecoder,
}

__all__ = [
    "BPDecoder",
    "PyMatchingDecoder",
    "UnionFindDecoder",
    "UnweightedUnionFindDecoder",
    "REGISTRY",
]

