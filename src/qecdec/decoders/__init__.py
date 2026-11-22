"""Decoder implementations registered with the common harness.

All strategies consume detector syndromes and return logical flips, even though
their internal physics models range from graph clusters to soft messages.
"""

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

