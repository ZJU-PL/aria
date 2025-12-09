"""Definitions for abstract state types.
"""
from typing import Any, Dict

# pylint: disable=too-few-public-methods


class AbstractState:
    """Describes a collection of concrete states.
    The following method should be overridden by sub-classes:
        __le__
    The other methods take their values from __le__.
    """

    def __le__(self, rhs: Any) -> bool:
        """True if self is exactly a subset of rhs.
        NOTE that this definition means (not (a <= b)) does NOT imply a > b.
        Perhaps we should raise an exception when elements are uncomparable.
        ``Exactly" here means that:
        self <= rhs
        <->
        (domain.gamma_hat(self) -> domain.gamma_hat(rhs))
        """
        raise NotImplementedError

    def __ge__(self, rhs: Any) -> bool:
        """True if self is a superset of rhs.
        NOTE that this definition means (not (a >= b)) does NOT imply a < b.
        Perhaps we should raise an exception when elements are uncomparable.
        """
        return rhs <= self

    def __eq__(self, rhs: Any) -> bool:
        """Determines if self and rhs represent the same AbstractState.
        """
        return self <= rhs <= self

    def __ne__(self, rhs: Any) -> bool:
        """Determines if self and rhs represent different AbstractStates.
        """
        return not self == rhs

    def translate(self, translation: Dict[str, str]) -> 'AbstractState':
        """Rename variables in the abstract state.
        Used in frontend/program.py to deal with "primes," i.e. we might encode
        x += y as x' = x + y, y' = y, but the interface we want is that the
        final abstract state should be written in terms of x and y. So we
        translate the abstract state in terms of x', y' by the translation
        dictionary {"x'": "x", "y'": "y"}.
        """
        raise NotImplementedError
