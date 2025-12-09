"""
Lightweight enum definitions used across the UFBV solver.

These enums are intentionally isolated to avoid circular imports and to keep
type names centralized and self-documented.
"""

from enum import Enum


class Quantification(Enum):
    """Quantifier kind at a position in a formula.

    - UNIVERSAL: universally quantified variables (over-approximation target)
    - EXISTENTIAL: existentially quantified variables (under-approximation target)
    """

    UNIVERSAL = 0
    EXISTENTIAL = 1


class Polarity(Enum):
    """Polarity of the current subformula during traversal.

    Polarity flips under negation and the antecedent of implication.
    """

    POSITIVE = 0
    NEGATIVE = 1


class ReductionType(Enum):
    """Bit-vector reduction (projection) strategy.

    Positive values represent left-extensions; negative values represent
    right-extensions. The sign is used to alternate directions when exploring
    a sequence of approximations.
    """

    ZERO_EXTENSION = 3
    ONE_EXTENSION = 1
    SIGN_EXTENSION = 2
    RIGHT_ZERO_EXTENSION = -3
    RIGHT_ONE_EXTENSION = -1
    RIGHT_SIGN_EXTENSION = -2


__all__ = [
    "Quantification",
    "Polarity",
    "ReductionType",
]
