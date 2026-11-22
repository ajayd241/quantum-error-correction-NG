"""Decoder implementations available to the common harness."""

from qecdec.base import Decoder
from qecdec.decoders.belief_find import BeliefFindDecoder
from qecdec.decoders.bp import BPDecoder
from qecdec.decoders.pymatching_ref import PyMatchingDecoder
from qecdec.decoders.union_find import (
    UnionFindDecoder,
    UnweightedUnionFindDecoder,
)


REGISTRY: dict[str, type[Decoder]] = {
    BeliefFindDecoder.name: BeliefFindDecoder,
    BPDecoder.name: BPDecoder,
    PyMatchingDecoder.name: PyMatchingDecoder,
    UnionFindDecoder.name: UnionFindDecoder,
    UnweightedUnionFindDecoder.name: UnweightedUnionFindDecoder,
}

__all__ = [
    "BeliefFindDecoder",
    "BPDecoder",
    "PyMatchingDecoder",
    "UnionFindDecoder",
    "UnweightedUnionFindDecoder",
    "REGISTRY",
]

