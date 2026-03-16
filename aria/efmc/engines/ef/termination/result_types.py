"""
Result types for termination verification.
"""

from typing import Optional
import z3


class TerminationResult:
    """
    Simple result class for termination verification.
    """

    def __init__(self, result: bool, time: float = 0.0, error: str = None):
        self.result = result
        self.time = time
        self.error = error


class NonTerminationResult:
    """
    Result class for non-termination verification using recurrence sets.
    """

    def __init__(
        self,
        result: bool,
        recurrence_set: Optional[z3.ExprRef] = None,
        time: float = 0.0,
        error: str = None,
    ):
        self.result = result  # True if non-termination is proven
        self.recurrence_set = recurrence_set  # The synthesized recurrence set
        self.time = time
        self.error = error
