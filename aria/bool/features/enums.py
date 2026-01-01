"""
Enumerations for variable and clause states in SAT solving.
"""

from enum import Enum, auto


class VarState(Enum):
    """Variable states in SAT solving."""

    TRUE_VAL = auto()
    FALSE_VAL = auto()
    UNASSIGNED = auto()
    IRRELEVANT = auto()


class ClauseState(Enum):
    """Clause states in SAT solving."""

    ACTIVE = auto()
    PASSIVE = auto()
