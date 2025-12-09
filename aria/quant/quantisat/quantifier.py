from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import sympy as sp


class Quantifier(ABC):
    """
    The abstract class for quantifiers.
    """

    @abstractmethod
    def negate(self) -> Quantifier:
        """
        Negate the quantified expression.

        Returns
        -------
        Quantifier
            The negated quantified expression.
        """
        pass

    @abstractmethod
    def subs(self, *args, **kwargs) -> Quantifier:
        """
        Substitute the variables in the ground expression.

        Returns
        -------
        Quantifier
            The quantifier with the substituted ground expression.
        """
        pass

    @abstractmethod
    def copy(self) -> Quantifier:
        """
        Copy the quantifier.

        Returns
        -------
        Quantifier
            The copied quantifier.
        """
        pass

    @abstractmethod
    def count_quantified_vars(self) -> Tuple[int, int]:
        """
        Count the number of quantified variables.

        Returns
        -------
        Tuple[int, int]
            The number of forall and exists variables.
        """
        pass

    @abstractmethod
    def count_quantifier_depth(self) -> int:
        """
        Count the quantifier depth.

        Returns
        -------
        int
            The quantifier depth.
        """
        pass

    @property
    @abstractmethod
    def free_symbols(self):
        """
        Get the free symbols in the quantifier.

        Returns
        -------
        List[sp.Symbol]
            The free symbols.
        """
        pass

    @abstractmethod
    def set_ground_formula(self, value):
        """
        Set a new ground formula.
        """
        pass


class ForAll(Quantifier):
    """
    The forall quantifier class.
    """

    def __init__(self, variables: List[sp.Symbol], formula: Union[sp.Basic, Quantifier]):
        """
        Initialize the forall quantifier.

        Parameters
        ----------
        variables : List[sp.Symbol]
            The quantified variables.
        formula : Union[sp.Basic, Quantifier]
            The formula to be quantified.
        """
        self.variables: List[sp.Symbol] = variables
        self.formula: Union[sp.Basic, Quantifier] = formula

    def __repr__(self):
        return f'Forall {", ".join([v.name for v in self.variables])}: {self.formula}'

    def negate(self):
        if isinstance(self.formula, Quantifier):
            return Exists(self.variables, self.formula.negate())
        else:
            return Exists(self.variables, sp.Not(self.formula))

    def subs(self, *args, **kwargs):
        return ForAll(self.variables, self.formula.subs(*args, **kwargs))

    def copy(self):
        try:
            return ForAll(self.variables, self.formula.copy())
        except AttributeError:
            return ForAll(self.variables, self.formula)

    def count_quantified_vars(self):
        if isinstance(self.formula, Quantifier):
            num_forall_vars, num_exists_vars = self.formula.count_quantified_vars()
            return len(self.variables) + num_forall_vars, num_exists_vars
        else:
            return len(self.variables), 0

    def count_quantifier_depth(self):
        if isinstance(self.formula, Exists):
            return 1 + self.formula.count_quantifier_depth()
        elif isinstance(self.formula, ForAll):
            return self.formula.count_quantifier_depth()
        else:
            return 1

    @property
    def free_symbols(self):
        result = []
        for s in self.formula.free_symbols:
            if s.name not in [v.name for v in self.variables]:
                result.append(s)

        return result

    def set_ground_formula(self, value):
        if isinstance(self.formula, Quantifier):
            self.formula.set_ground_formula(value)
        else:
            self.formula = value


class Exists(Quantifier):
    """
    The exists quantifier class.
    """

    def __init__(self, variables: List[sp.Symbol], formula: Union[sp.Basic, Quantifier]):
        """
        Initialize the exists quantifier.

        Parameters
        ----------
        variables : List[sp.Symbol]
            The quantified variables.
        formula : Union[sp.Basic, Quantifier]
            The formula to be quantified.
        """
        self.variables: List[sp.Symbol] = variables
        self.formula: Union[sp.Basic, Quantifier] = formula

    def __repr__(self):
        return f'Exists {", ".join([v.name for v in self.variables])}: {self.formula}'

    def negate(self):
        if isinstance(self.formula, Quantifier):
            return ForAll(self.variables, self.formula.negate())
        else:
            return ForAll(self.variables, sp.Not(self.formula))

    def subs(self, *args, **kwargs):
        return Exists(self.variables, self.formula.subs(*args, **kwargs))

    def copy(self):
        try:
            return Exists(self.variables, self.formula.copy())
        except AttributeError:
            return Exists(self.variables, self.formula)

    def count_quantified_vars(self):
        if isinstance(self.formula, Quantifier):
            num_forall_vars, num_exists_vars = self.formula.count_quantified_vars()
            return num_forall_vars, len(self.variables) + num_exists_vars
        else:
            return 0, len(self.variables)

    def count_quantifier_depth(self):
        if isinstance(self.formula, ForAll):
            return 1 + self.formula.count_quantifier_depth()
        elif isinstance(self.formula, Exists):
            return self.formula.count_quantifier_depth()
        else:
            return 1

    @property
    def free_symbols(self):
        result = []
        for s in self.formula.free_symbols:
            if s.name not in [v.name for v in self.variables]:
                result.append(s)

        return result

    def set_ground_formula(self, value):
        if isinstance(self.formula, Quantifier):
            self.formula.set_ground_formula(value)
        else:
            self.formula = value
