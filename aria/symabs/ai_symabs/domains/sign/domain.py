"""Main class definition for the Signs conjunctive domain.
"""
from typing import Dict, List, Optional
import z3
from ..z3_variables import Z3VariablesDomain
from ..z3_variables.concrete import Z3VariablesState
from .abstract import Sign, SignAbstractState


class SignDomain(Z3VariablesDomain):
    """Represents an abstract space over the sign of variables.
    """

    def __init__(self, variables: List[str]) -> None:
        """Constructs a new SignDomain, with variables named in variables.

        variables should be a list of variable names
        """
        Z3VariablesDomain.__init__(self, variables, z3.Int)

    def gamma_hat(self, alpha: SignAbstractState) -> z3.BoolRef:
        """Returns a formula describing the same states as alpha
        """
        conjunctions: List[z3.BoolRef] = []
        for name in self.variables:
            sign = alpha.sign_of(name)
            if sign == Sign.Positive:
                conjunctions.append(self.z3_variable(name) > 0)
            elif sign == Sign.Negative:
                conjunctions.append(self.z3_variable(name) < 0)
            elif sign == Sign.Bottom:
                conjunctions.append(self.z3_variable(name) !=
                                    self.z3_variable(name))
            elif sign == Sign.Top:
                conjunctions.append(self.z3_variable(name) ==
                                    self.z3_variable(name))
        return z3.And(*conjunctions)

    def join(self, elements: List[SignAbstractState]) -> SignAbstractState:
        """Returns the join of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the smallest alpha
        containing all alpha_1, ..., alpha_n. It may not be in elements.
        """
        joined = self.bottom
        for state in elements:
            for name in self.variables:
                sign1 = joined.sign_of(name)
                sign2 = state.sign_of(name)
                if sign1 <= sign2:
                    joined.set_sign(name, sign2)
                elif sign2 <= sign1:
                    joined.set_sign(name, sign1)
                else:
                    # In the sign domain, this is only true if sign1 and sign2
                    # are opposite Positive/Negative, so Top is the only join
                    joined.set_sign(name, Sign.Top)
        return joined

    def meet(self, elements: List[SignAbstractState]) -> SignAbstractState:
        """Returns the meet of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the greatest alpha
        contained by all alpha_1, ..., alpha_n. It may not be in elements.
        """
        met = self.top
        for state in elements:
            for name in self.variables:
                sign1 = met.sign_of(name)
                sign2 = state.sign_of(name)
                if sign1 <= sign2:
                    met.set_sign(name, sign1)
                elif sign2 <= sign1:
                    met.set_sign(name, sign2)
                else:
                    # In the sign domain, this is only true if sign1 and sign2
                    # are opposite Positive/Negative, so Bottom is the only meet
                    met.set_sign(name, Sign.Bottom)
        return met

    def abstract_consequence(self, lower: SignAbstractState, upper: SignAbstractState) -> SignAbstractState:
        """Returns the "abstract consequence" of lower and upper.

        The abstract consequence must be a superset of lower and *NOT* a
        superset of upper.

        Note that this is a fairly "simple" abstract consequence, in that it
        sets only one variable to a non-top sign. This improves performance of
        the SMT solver in many cases. In certain cases, other choices for the
        abstract consequence will lead to better algorithm performance.
        """
        for variable in self.variables:
            proposed = self.top.copy()
            proposed.set_sign(variable, lower.sign_of(variable))
            if not proposed >= upper:
                return proposed
        return lower.copy()

    # Converts one concrete set of variables into an abstract element
    def beta(self, sigma: Z3VariablesState) -> SignAbstractState:
        """Returns the least abstract state describing sigma.

        Sigma should be an Z3VariablesState. See Definition 3.4 in:
        Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
        Applications (Ph.D. dissertation). Computer Sciences Department,
        University of Wisconsin, Madison.
        """
        return SignAbstractState(
            dict((name, Sign.from_number(sigma.value_of(name)))
                 for name in self.variables))

    @property
    def top(self) -> SignAbstractState:
        """Returns the least upper bound of the entire abstract space.
        """
        return SignAbstractState(dict((name, Sign.Top)
                                      for name in self.variables))

    @property
    def bottom(self) -> SignAbstractState:
        """Returns the greatest lower bound of the entire abstract space.
        """
        return SignAbstractState(dict((name, Sign.Bottom)
                                      for name in self.variables))
