"""Sequence metrics used by the ONOTE CNC and AST evaluators."""

from difflib import SequenceMatcher
from typing import Sequence, TypeVar


T = TypeVar("T")


def matching_block_count(reference: Sequence[T], prediction: Sequence[T]) -> int:
    """Return the total exact matching-block length from SequenceMatcher."""
    if not reference or not prediction:
        return 0
    matcher = SequenceMatcher(None, reference, prediction)
    return sum(block.size for block in matcher.get_matching_blocks())


def sequence_similarity(reference: Sequence[T], prediction: Sequence[T]) -> float:
    """Return the SequenceMatcher ratio (SR) as a value in [0, 1]."""
    if not reference or not prediction:
        return 0.0
    matched = matching_block_count(reference, prediction)
    return 2.0 * matched / (len(reference) + len(prediction))


def matching_precision(reference: Sequence[T], prediction: Sequence[T]) -> float:
    """Return matching-block precision (MP), normalized by prediction length."""
    if not prediction:
        return 0.0
    return matching_block_count(reference, prediction) / len(prediction)
