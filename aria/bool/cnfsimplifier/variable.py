"""
Variable
"""
from typing import Union


class Variable:
    """
    A Boolean variable
    """
    variable_value: int  # 变量值
    variable_abs: int   # 变量绝对值
    signal: int        # 变量符号(+1/-1)

    def __init__(self, variable: int) -> None:
        self.variable_value = variable
        self.variable_abs = abs(self.variable_value)
        self.signal = 1
        if self.variable_value < 0:
            self.signal = -1

    def copy(self) -> 'Variable':
        """
        Copy the variable
        """
        return Variable(self.variable_value)

    def __str__(self) -> str:
        """
        String representation of the variable
        """
        return f"Variable({self.variable_value})"

    def __repr__(self) -> str:
        """
        Representation of the variable
        """
        return self.__str__()

    def __eq__(self, other: 'Variable') -> bool:
        """
        Equality of the variable
        """
        return self.variable_value == other.variable_value

    def __ne__(self, other: 'Variable') -> bool:
        """
        Inequality of the variable
        """
        return self.variable_value != other.variable_value

    def __lt__(self, other: 'Variable') -> bool:
        """
        Less than of the variable
        """
        return self.variable_value < other.variable_value

    def __le__(self, other: 'Variable') -> bool:
        """
        Less than or equal to of the variable
        """
        return self.variable_value <= other.variable_value

    def __gt__(self, other: 'Variable') -> bool:
        """
        Greater than of the variable
        """
        return self.variable_value > other.variable_value

    def __ge__(self, other: 'Variable') -> bool:
        """
        Greater than or equal to of the variable
        """
        return self.variable_value >= other.variable_value

    def __neg__(self) -> 'Variable':
        """
        Negation of the variable
        """
        return Variable(-self.variable_value)

    def __pos__(self) -> 'Variable':
        """
        Positive of the variable
        """
        return self

    def __abs__(self) -> 'Variable':
        """
        Absolute value of the variable
        """
        return Variable(abs(self.variable_value))

    def __int__(self) -> int:
        """
        Integer value of the variable
        """
        return int(self.variable_value)

    def __float__(self) -> float:
        """
        Float value of the variable
        """
        return float(self.variable_value)

    def __complex__(self):
        """
        Complex value of the variable
        """
        return complex(self.variable_value)

    def __hash__(self):
        """
        Hash of the variable
        """
        return hash(self.variable_value)
