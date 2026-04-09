"""Small reusable reified Boolean gadgets."""

from typing import List, Sequence


def implies(left: int, right: int) -> List[List[int]]:
    """Encode ``left -> right``."""

    return [[-int(left), int(right)]]


def equivalent(left: int, right: int) -> List[List[int]]:
    """Encode ``left <-> right``."""

    return [[-int(left), int(right)], [-int(right), int(left)]]


def reify_and(output: int, inputs: Sequence[int]) -> List[List[int]]:
    """Encode ``output <-> and(inputs)``."""

    out = int(output)
    normalized = [int(value) for value in inputs]
    clauses = [[-out, literal] for literal in normalized]
    clauses.append([out] + [-literal for literal in normalized])
    return clauses


def reify_or(output: int, inputs: Sequence[int]) -> List[List[int]]:
    """Encode ``output <-> or(inputs)``."""

    out = int(output)
    normalized = [int(value) for value in inputs]
    clauses = [[-out] + normalized]
    clauses.extend([[out, -literal] for literal in normalized])
    return clauses
