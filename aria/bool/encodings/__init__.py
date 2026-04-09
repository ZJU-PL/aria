"""Reusable Boolean encoding helpers."""

from .cardinality import (
    EncodingResult,
    at_least_k,
    at_least_k_sequential_counter,
    at_least_k_totalizer,
    at_most_k,
    at_most_k_sequential_counter,
    at_most_k_totalizer,
    at_most_one,
    exactly_k,
    exactly_k_sequential_counter,
    exactly_k_totalizer,
    exactly_one,
    pb_at_least,
    pb_at_most,
    pb_equals,
)
from .reify import equivalent, implies, reify_and, reify_or


def encode_at_least_one(literals):
    """Compatibility wrapper for a single at-least-one clause."""

    normalized = [int(literal) for literal in literals]
    return EncodingResult(
        clauses=[normalized], top_id=max([0] + [abs(literal) for literal in normalized])
    )


def encode_at_most_one_pairwise(literals):
    """Compatibility wrapper for pairwise at-most-one encoding."""

    normalized = [int(literal) for literal in literals]
    clauses = []
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            clauses.append([-left, -right])
    return EncodingResult(
        clauses=clauses, top_id=max([0] + [abs(literal) for literal in normalized])
    )


def encode_exactly_one_pairwise(literals):
    """Compatibility wrapper for pairwise exactly-one encoding."""

    atleast = encode_at_least_one(literals)
    atmost = encode_at_most_one_pairwise(literals)
    return EncodingResult(
        clauses=atleast.clauses + atmost.clauses,
        top_id=max(atleast.top_id, atmost.top_id),
    )


__all__ = [
    "EncodingResult",
    "at_most_k",
    "at_least_k",
    "exactly_k",
    "at_most_one",
    "exactly_one",
    "at_most_k_totalizer",
    "at_least_k_totalizer",
    "exactly_k_totalizer",
    "at_most_k_sequential_counter",
    "at_least_k_sequential_counter",
    "exactly_k_sequential_counter",
    "pb_at_most",
    "pb_at_least",
    "pb_equals",
    "implies",
    "equivalent",
    "reify_and",
    "reify_or",
    "encode_at_least_one",
    "encode_at_most_one_pairwise",
    "encode_exactly_one_pairwise",
]
