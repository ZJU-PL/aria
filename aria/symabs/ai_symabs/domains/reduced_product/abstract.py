"""Definitions for the ReducedProduct abstract states.
"""
from typing import Any, Dict
from ..core.abstract import AbstractState


# TODO(masotoud): find a different naming scheme that makes this clearer, if
# possible.
# pylint: disable=invalid-name


class ReducedProductAbstractState(AbstractState):
    """Abstract state describing the signs of a collection of variables.
    """

    def __init__(self, state_A: AbstractState, state_B: AbstractState) -> None:
        """Construct a new ReducedProductAbstractState.

        @state_A should be an AbstractState in the first domain while @state_B
        should be a state in the second domain. They should both describe the
        same set of variables.
        """
        self.state_A: AbstractState = state_A
        self.state_B: AbstractState = state_B

    def copy(self) -> 'ReducedProductAbstractState':
        """A new ReducedProductAbstractState representing the same state.
        """
        return ReducedProductAbstractState(self.state_A.copy(),
                                           self.state_B.copy())

    def __le__(self, rhs: 'ReducedProductAbstractState') -> bool:
        """True if self represents a subset of rhs.

        Note that this definition means (not (a <= b)) does NOT imply a > b.
        Perhaps we should raise an exception when elements are uncomparable.
        This assumes that both have the exact same set of variables, and does
        not check that condition.
        """
        return self.state_A <= rhs.state_A and self.state_B <= rhs.state_B

    def translate(self, translation: Dict[str, str]) -> 'ReducedProductAbstractState':
        """Rename the variables in the abstract state.
        """
        return ReducedProductAbstractState(self.state_A.translate(translation),
                                           self.state_B.translate(translation))

    def __str__(self) -> str:
        """Human-readable form of the abstract state.
        """
        return str(self.state_A) + "\n" + str(self.state_B)
