"""Reusable Boolean cardinality and lightweight PB encodings."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from pysat.card import CardEnc, EncType
from pysat.formula import CNF


_ENCODINGS: Dict[str, int] = {
    "pairwise": EncType.pairwise,
    "seqcounter": EncType.seqcounter,
    "sortnetwrk": EncType.sortnetwrk,
    "cardnetwrk": EncType.cardnetwrk,
    "bitwise": EncType.bitwise,
    "ladder": EncType.ladder,
    "totalizer": EncType.totalizer,
    "mtotalizer": EncType.mtotalizer,
    "kmtotalizer": EncType.kmtotalizer,
}


@dataclass(frozen=True)
class EncodingResult:
    """CNF clauses emitted by an encoding."""

    clauses: List[List[int]]
    top_id: int

    def to_cnf(self) -> CNF:
        """Convert the emitted clauses to a PySAT CNF object."""

        return CNF(from_clauses=self.clauses)


def _normalize_literals(literals: Sequence[int]) -> List[int]:
    return [int(literal) for literal in literals]


def _resolve_encoding(encoding: str) -> int:
    try:
        return _ENCODINGS[encoding]
    except KeyError as error:
        raise ValueError("unsupported cardinality encoding '{}'".format(encoding)) from error


def _max_var(literals: Sequence[int], top_id: int) -> int:
    return max([top_id] + [abs(literal) for literal in literals])


def _normalize_weighted_terms(
    literals: Sequence[int], weights: Sequence[int]
) -> Tuple[List[int], List[int]]:
    if len(literals) != len(weights):
        raise ValueError("literals and weights must have the same length")

    normalized_literals: List[int] = []
    normalized_weights: List[int] = []
    for literal, weight in zip(literals, weights):
        int_weight = int(weight)
        if int_weight < 0:
            raise ValueError("pseudo-Boolean weights must be non-negative")
        if int_weight == 0:
            continue
        normalized_literals.append(int(literal))
        normalized_weights.append(int_weight)
    return normalized_literals, normalized_weights


def _expand_weighted_literals(
    literals: Sequence[int], weights: Sequence[int]
) -> List[int]:
    expanded: List[int] = []
    for literal, weight in zip(literals, weights):
        expanded.extend([int(literal)] * int(weight))
    return expanded


def at_most_k(
    literals: Sequence[int], bound: int, top_id: int = 0, encoding: str = "seqcounter"
) -> EncodingResult:
    """Encode an at-most-k constraint."""

    normalized = _normalize_literals(literals)
    if bound < 0:
        return EncodingResult(clauses=[[]], top_id=_max_var(normalized, top_id))
    if bound >= len(normalized):
        return EncodingResult(clauses=[], top_id=_max_var(normalized, top_id))
    encoded = CardEnc.atmost(
        lits=normalized,
        bound=bound,
        top_id=_max_var(normalized, top_id),
        encoding=_resolve_encoding(encoding),
    )
    return EncodingResult(clauses=list(encoded.clauses), top_id=encoded.nv)


def at_least_k(
    literals: Sequence[int], bound: int, top_id: int = 0, encoding: str = "seqcounter"
) -> EncodingResult:
    """Encode an at-least-k constraint."""

    normalized = _normalize_literals(literals)
    if bound <= 0:
        return EncodingResult(clauses=[], top_id=_max_var(normalized, top_id))
    if bound > len(normalized):
        return EncodingResult(clauses=[[]], top_id=_max_var(normalized, top_id))
    encoded = CardEnc.atleast(
        lits=normalized,
        bound=bound,
        top_id=_max_var(normalized, top_id),
        encoding=_resolve_encoding(encoding),
    )
    return EncodingResult(clauses=list(encoded.clauses), top_id=encoded.nv)


def exactly_k(
    literals: Sequence[int], bound: int, top_id: int = 0, encoding: str = "seqcounter"
) -> EncodingResult:
    """Encode an exactly-k constraint."""

    normalized = _normalize_literals(literals)
    if not normalized and bound == 0:
        return EncodingResult(clauses=[], top_id=top_id)
    encoded = CardEnc.equals(
        lits=normalized,
        bound=bound,
        top_id=_max_var(normalized, top_id),
        encoding=_resolve_encoding(encoding),
    )
    return EncodingResult(clauses=list(encoded.clauses), top_id=encoded.nv)


def at_most_one(
    literals: Sequence[int], top_id: int = 0, encoding: str = "seqcounter"
) -> EncodingResult:
    """Encode an at-most-one constraint."""

    return at_most_k(literals, 1, top_id=top_id, encoding=encoding)


def exactly_one(
    literals: Sequence[int], top_id: int = 0, encoding: str = "seqcounter"
) -> EncodingResult:
    """Encode an exactly-one constraint."""

    return exactly_k(literals, 1, top_id=top_id, encoding=encoding)


def at_most_k_totalizer(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an at-most-k constraint with a totalizer."""

    return at_most_k(literals, bound, top_id=top_id, encoding="totalizer")


def at_least_k_totalizer(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an at-least-k constraint with a totalizer."""

    return at_least_k(literals, bound, top_id=top_id, encoding="totalizer")


def exactly_k_totalizer(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an exactly-k constraint with a totalizer."""

    return exactly_k(literals, bound, top_id=top_id, encoding="totalizer")


def at_most_k_sequential_counter(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an at-most-k constraint with a sequential counter."""

    return at_most_k(literals, bound, top_id=top_id, encoding="seqcounter")


def at_least_k_sequential_counter(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an at-least-k constraint with a sequential counter."""

    return at_least_k(literals, bound, top_id=top_id, encoding="seqcounter")


def exactly_k_sequential_counter(
    literals: Sequence[int], bound: int, top_id: int = 0
) -> EncodingResult:
    """Encode an exactly-k constraint with a sequential counter."""

    return exactly_k(literals, bound, top_id=top_id, encoding="seqcounter")


def pb_at_most(
    literals: Sequence[int],
    weights: Sequence[int],
    bound: int,
    top_id: int = 0,
    encoding: str = "totalizer",
) -> EncodingResult:
    """Encode a non-negative pseudo-Boolean upper bound.

    This lightweight implementation expands integral weights into a
    cardinality constraint. It is dependency-free and works without `pypblib`.
    """

    normalized_literals, normalized_weights = _normalize_weighted_terms(
        literals, weights
    )
    expanded = _expand_weighted_literals(normalized_literals, normalized_weights)
    return at_most_k(expanded, bound, top_id=top_id, encoding=encoding)


def pb_at_least(
    literals: Sequence[int],
    weights: Sequence[int],
    bound: int,
    top_id: int = 0,
    encoding: str = "totalizer",
) -> EncodingResult:
    """Encode a non-negative pseudo-Boolean lower bound."""

    normalized_literals, normalized_weights = _normalize_weighted_terms(
        literals, weights
    )
    expanded = _expand_weighted_literals(normalized_literals, normalized_weights)
    return at_least_k(expanded, bound, top_id=top_id, encoding=encoding)


def pb_equals(
    literals: Sequence[int],
    weights: Sequence[int],
    bound: int,
    top_id: int = 0,
    encoding: str = "totalizer",
) -> EncodingResult:
    """Encode a non-negative pseudo-Boolean equality."""

    normalized_literals, normalized_weights = _normalize_weighted_terms(
        literals, weights
    )
    expanded = _expand_weighted_literals(normalized_literals, normalized_weights)
    return exactly_k(expanded, bound, top_id=top_id, encoding=encoding)
