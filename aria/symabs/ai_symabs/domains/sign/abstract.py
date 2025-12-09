"""Definitions for the Signs abstract space.
"""
from enum import Enum
from typing import Any, Dict
from ..core.abstract import AbstractState


# pylint: disable=too-few-public-methods
class Sign(Enum):
    """Represents the sign of a variable
    """

    Bottom = 1
    Negative = 2
    Positive = 3
    Top = 4

    def __le__(self, rhs: 'Sign') -> bool:
        """Returns True if self represents a subset of rhs.
        """
        return (self == rhs
                or rhs is Sign.Top
                or self is Sign.Bottom)

    @staticmethod
    def from_number(number: float) -> 'Sign':
        """Returns the most precise Sign describing number
        """
        if number > 0:
            return Sign.Positive
        if number < 0:
            return Sign.Negative
        return Sign.Top


class SignAbstractState(AbstractState):
    """Abstract state describing the signs of a collection of variables.
    """

    def __init__(self, variable_signs: Dict[str, Sign]) -> None:
        """Initializer for the SignAbstractState.

        variable_signs should be a dictionary of { "variable_name": sign },
        where sign is an element of the Sign enum.
        """
        self.variable_signs: Dict[str, Sign] = variable_signs
        self.variables: list[str] = list(variable_signs.keys())

    def copy(self) -> 'SignAbstractState':
        """Returns a new SignAbstractState with the same signs as self.
        """
        return SignAbstractState(self.variable_signs.copy())

    def sign_of(self, variable_name: str) -> Sign:
        """Returns the sign of a variable given its name.

        Sign will be Sign enums.
        """
        return self.variable_signs[variable_name]

    def set_sign(self, variable_name: str, sign: Sign) -> None:
        """Sets the sign of a variable given its name.
        """
        self.variable_signs[variable_name] = sign

    def __le__(self, rhs: 'SignAbstractState') -> bool:
        """True if self represents a subset of rhs.

        Note that this definition means (not (a <= b)) does NOT imply a > b.
        Perhaps we should raise an exception when elements are uncomparable.
        This assumes that both have the exact same set of variables, and does
        not check that condition.
        """
        return all(self.sign_of(name) <= rhs.sign_of(name)
                   for name in self.variables)

    def translate(self, translation: Dict[str, str]) -> 'SignAbstractState':
        return SignAbstractState({
            translation[name]: state
            for name, state in self.variable_signs.items()
        })

    def __str__(self) -> str:
        return str(self.variable_signs)
