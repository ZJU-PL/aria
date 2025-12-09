"""Main class definition for the Intervals conjunctive domain.
"""
from typing import Any, List
import z3
from ..z3_variables import Z3VariablesDomain
from .abstract import Interval, IntervalAbstractState


class IntervalDomain(Z3VariablesDomain):
    """Represents an abstract space over the intervals of variables.
    """

    def __init__(self, variables: List[str]) -> None:
        """Constructs a new IntervalDomain, with variables named in variables.

        variables should be a list of variable names
        """
        Z3VariablesDomain.__init__(self, variables, z3.Int)

    def gamma_hat(self, alpha: IntervalAbstractState) -> Any:
        """Returns a formula describing the same states as alpha
        """
        conjunctions = []
        for name in self.variables:
            interval = alpha.interval_of(name)
            if isinstance(interval.lower, int):
                conjunctions.append(interval.lower <= self.z3_variable(name))
            elif interval.lower == float("inf"):
                conjunctions.append(False)
            if isinstance(interval.upper, int):
                conjunctions.append(self.z3_variable(name) <= interval.upper)
            elif interval.upper == float("-inf"):
                conjunctions.append(False)
        return z3.And(*conjunctions)

    def join(self, elements: List[IntervalAbstractState]) -> IntervalAbstractState:
        """Returns the join of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the smallest alpha
        containing all alpha_1, ..., alpha_n. It may not be in elements.
        """
        joined = self.bottom
        for state in elements:
            for name in self.variables:
                joined_interval = joined.interval_of(name)
                state_interval = state.interval_of(name)
                union = joined_interval.union(state_interval)
                joined.set_interval(name, union)
        return joined

    def meet(self, elements: List[IntervalAbstractState]) -> IntervalAbstractState:
        """Returns the meet of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the greatest alpha
        contained by all alpha_1, ..., alpha_n. It may not be in elements.
        """
        met = self.top
        for state in elements:
            for name in self.variables:
                met_interval = met.interval_of(name)
                state_interval = state.interval_of(name)
                intersection = met_interval.intersection(state_interval)
                met.set_interval(name, intersection)
        return met

    def abstract_consequence(self, lower: IntervalAbstractState, upper: IntervalAbstractState) -> IntervalAbstractState:
        """Returns the "abstract consequence" of lower and upper.

        The abstract consequence must be a superset of lower and *NOT* a
        superset of upper.

        Note that this is a fairly "simple" abstract consequence, in that it
        sets only one variable to a non-top interval. This improves performance
        of the SMT solver in many cases. In certain cases, other choices for
        the abstract consequence will lead to better algorithm performance.
        """
        for variable in self.variables:
            proposed = self.top.copy()
            proposed.set_interval(variable, lower.interval_of(variable))
            if not proposed >= upper:
                return proposed
        return lower.copy()

    # Converts one concrete set of variables into an abstract element
    def beta(self, sigma: Any) -> IntervalAbstractState:
        """Returns the least abstract state describing sigma.

        Sigma should be an Z3VariablesState. See Definition 3.4 in:
        Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
        Applications (Ph.D. dissertation). Computer Sciences Department,
        University of Wisconsin, Madison.
        """
        return IntervalAbstractState(
            dict({name: Interval(sigma.value_of(name), sigma.value_of(name))
                  for name in self.variables}))

    @property
    def top(self) -> IntervalAbstractState:
        """Returns the least upper bound of the entire abstract space.
        """
        top_interval = Interval(float("-inf"), float("inf"))
        return IntervalAbstractState({name: top_interval for name in self.variables})

    @property
    def bottom(self) -> IntervalAbstractState:
        """Returns the greatest lower bound of the entire abstract space.
        """
        bottom_interval = Interval(float("inf"), float("-inf"))
        return IntervalAbstractState({name: bottom_interval for name in self.variables})
