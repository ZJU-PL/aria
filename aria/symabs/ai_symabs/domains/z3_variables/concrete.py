"""Definitions for the Z3 Variables concrete space.
"""
from typing import Any, Dict, Union
from ..core.concrete import ConcreteState


# pylint: disable=too-few-public-methods
class Z3VariablesState(ConcreteState):
    """Describes a single concrete state of the program.
    Here we assume there are some variables with certain values.
    """

    # pylint: disable=unused-argument
    def __init__(self, variable_values: Dict[str, Union[int, float]], variable_type: Any = None) -> None:
        """Constructs a new Z3VariablesState.

        variable_values should be a dictionary of form { "variable_name": value
        }, where value is an integer.

        variable_type is not currently used, but is provided for consistency
        with Z3VariablesState
        """
        self.variable_values: Dict[str, Union[int, float]] = variable_values

    def value_of(self, name: str) -> Union[int, float]:
        """Returns the integer value of a given variable.
        """
        return self.variable_values[name]

    def __repr__(self) -> str:
        """Returns a human-readable representation of self
        """
        return ("{" +
                ", ".join(f"{name}: {value}"
                          for name, value
                          in sorted(self.variable_values.items()))
                + "}")
