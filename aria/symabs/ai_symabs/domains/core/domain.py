"""Definitions for abstract domain types.
"""
from typing import Any, List, Dict

class ConjunctiveDomain:
    """Represents a conjunctive domain.
    See Definition 3.5 in:
    Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
    Applications (Ph.D. dissertation). Computer Sciences Department, University
    of Wisconsin, Madison.
    """

    def model(self, phi: Any) -> Any:
        """Returns a solution to logical formula phi.
        The satisfying model is returned as a ConcreteElement, or None if the
        model is unsatisfyable.
        """
        raise NotImplementedError

    def model_and(self, phi1: Any, phi2: Any) -> Any:
        """Returns a solution to logical formula phi1 && phi2.
        The satisfying model is returned as a ConcreteElement, or None if the
        model is unsatisfyable.
        This is used in Z3 domains to automatically use the iterative solver
        whenever phi1 is the same between calls.
        """
        return self.model(self.logic_and([phi1, phi2]))

    def gamma_hat(self, alpha: Any) -> Any:
        """Translates abstract element alpha into a logical formula.
        See Definition 3.10 in:
        Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
        Applications (Ph.D. dissertation). Computer Sciences Department,
        University of Wisconsin, Madison.
        """
        raise NotImplementedError

    def logic_and(self, formulas: List[Any]) -> Any:
        """Returns the logical and of the given formulas.
        """
        raise NotImplementedError

    def logic_not(self, formula: Any) -> Any:
        """Returns the logical negation of the given formula.
        """
        raise NotImplementedError

    def join(self, elements: List[Any]) -> Any:
        """Returns the least-upper-bound for elements.
        Elements should be a list of AbstractElements. The existence of such an
        upper bound is guaranteed by Definition 3.1 for a complete lattice.
        """
        raise NotImplementedError

    def meet(self, elements: List[Any]) -> Any:
        """Returns the greatest-lower-bound for elements.
        Elements should be a list of AbstractElements. The existence of such a
        lower bound is guaranteed by Definition 3.1 for a complete lattice.
        """
        raise NotImplementedError

    def abstract_consequence(self, lower: Any, upper: Any) -> Any:
        """Returns the "abstract consequence" of lower and upper.
        The abstract consequence must be a superset of lower and *NOT* a
        superset of upper.
        """
        raise NotImplementedError

    def beta(self, sigma: Any) -> Any:
        """Returns the least abstract state which describes sigma.
        Sigma should be a ConcreteElement. See Definition 3.4 in:
        Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
        Applications (Ph.D. dissertation). Computer Sciences Department,
        University of Wisconsin, Madison.
        """
        raise NotImplementedError

    @property
    def top(self) -> Any:
        """Returns the least-upper-bound of the entire abstract space.
        Guaranteed by Definition 3.1
        """
        raise NotImplementedError

    @property
    def bottom(self) -> Any:
        """Returns the greatest-lower-bound of the entire abstract space.
        Guaranteed by Definition 3.1
        """
        raise NotImplementedError

    def translate(self, translation: Dict[str, str]) -> 'ConjunctiveDomain':
        """Rename variables in the abstract space definition.
        Used in frontend/program.py to deal with "primes." We might encode x +=
        y as x' = x + y, y' = y, but the user will give us a domain in terms of
        x and y (which implicitly should refer to x' and y' at the end of the
        program snippet). So we translate the abstract domain given by the user
        in terms of x, y by the translation dictionary {"x": "x'", "y": "y'"}.
        Note that we use AbstractState.translate(...) to translate back to the
        user's preferred naming (x, y).
        """
        raise NotImplementedError
