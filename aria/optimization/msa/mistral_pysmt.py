"""
Computing Minimal Satisfying Assignment using PySMT.

Based on the algorithm from Alessandro Previti and Alexey S. Ignatiev.
"""
import logging
from typing import Dict, FrozenSet, List, Optional, Set

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Bool, ForAll, Not, Solver, get_model, qelim
from pysmt.smtlib.parser import SmtLibParser

logger = logging.getLogger(__name__)


def get_qmodel(
    x_univl: Set,
    formula,
    maxiters: Optional[int] = None,
    solver_name: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """Simple 2QBF CEGAR (Counter-Example Guided Abstraction Refinement) for SMT.

    Args:
        x_univl: Set of universally quantified variables
        formula: PySMT formula to solve
        maxiters: Maximum number of iterations (None for unlimited)
        solver_name: Name of the solver to use
        verbose: If True, print debug information

    Returns:
        Dictionary mapping existential variables to their values, or None if unsat

    Raises:
        SolverReturnedUnknownResultError: If maxiters is reached without solution
    """
    x_univl_set = set(x_univl)
    x_exist = formula.get_free_variables() - x_univl_set

    with Solver(name=solver_name) as solver:
        solver.add_assertion(Bool(True))
        iteration = 0

        while maxiters is None or iteration <= maxiters:
            iteration += 1

            if not solver.solve():
                return None

            # Get candidate assignment for existential variables
            candidate = {v: solver.get_value(v) for v in x_exist}
            subformula = formula.substitute(candidate).simplify()
            if verbose:
                logger.debug("qsolve candidate %d: %s", iteration, candidate)

            # Check if candidate is valid for all universal assignments
            counter_model = get_model(Not(subformula), solver_name=solver_name)
            if counter_model is None:
                # No counter-example found, candidate is valid
                return candidate

            # Counter-example found, refine
            counter_example = {v: counter_model[v] for v in x_univl_set}
            subformula = formula.substitute(counter_example).simplify()
            if verbose:
                logger.debug("qsolve counter-example %d: %s", iteration, counter_example)

            solver.add_assertion(subformula)

        raise SolverReturnedUnknownResultError


class Mistral:
    """Mistral solver class for Minimal Satisfying Assignment using PySMT."""

    def __init__(
        self,
        simplify: bool,
        solver: str,
        qsolve: str,
        verbose: int,
        fname: str,
    ) -> None:
        """Initialize the Mistral solver.

        Args:
            simplify: Whether to simplify the formula
            solver: Name of the solver to use
            qsolve: Quantifier solving method ('std', 'z3qe', or 'cegar')
            verbose: Verbosity level (0=silent, 1=normal, 2+=debug)
            fname: Path to the SMT-LIB file
        """
        self.script = SmtLibParser().get_script_fname(fname)
        self.formula = self.script.get_last_formula()
        if simplify:
            self.formula = self.formula.simplify()
        self.fvars = self.formula.get_free_variables()

        self.cost = 0
        self.sname = solver
        self.verb = verbose
        self.qsolve = qsolve

        if self.verb > 2:
            logger.debug("Formula: %s", self.formula)

        if self.verb > 1:
            logger.debug("Variables (%d): %s", len(self.fvars), list(self.fvars))

    def solve(self) -> Optional[List[str]]:
        """Find minimal satisfying assignment.

        Implements the find_msa() procedure from Fig. 2 of the dillig-cav12 paper.

        Returns:
            List of variable assignments as strings, or None if unsatisfiable
        """
        # Test if formula is satisfiable
        if not get_model(self.formula, solver_name=self.sname):
            return None

        mus = self.compute_mus(frozenset([]), self.fvars, 0)
        model = self.get_model_forall(mus)
        if model is None:
            return None

        return [f'{v}={model[v]}' for v in self.fvars - mus]

    def compute_mus(
        self,
        X: FrozenSet,
        fvars: FrozenSet,
        lb: int,
    ) -> FrozenSet:
        """Compute Minimal Unsatisfiable Subset (MUS).

        Implements the find_mus() procedure from Fig. 1 of the dillig-cav12 paper.

        Args:
            X: Current set of universally quantified variables
            fvars: Remaining free variables to consider
            lb: Lower bound on the size of the MUS

        Returns:
            FrozenSet of variables in the MUS

        Note:
            Variable selection (x) could be improved with a smarter heuristic.
        """
        if not fvars or len(fvars) <= lb:
            return frozenset()

        best: FrozenSet = frozenset()
        # TODO: Choose x in a more clever way (e.g., based on variable importance)
        x = frozenset([next(iter(fvars))])

        if self.verb > 1:
            logger.debug("State: X = %s + %s, lb = %d", list(X), list(x), lb)

        if self.get_model_forall(X.union(x)):
            Y = self.compute_mus(X.union(x), fvars - x, lb - 1)

            cost_curr = len(Y) + 1
            if cost_curr > lb:
                best = Y.union(x)
                lb = cost_curr

        Y = self.compute_mus(X, fvars - x, lb)
        if len(Y) > lb:
            best = Y

        return best

    def get_model_forall(self, x_univl: FrozenSet):
        """Get a model for a universally quantified formula.

        Uses different quantifier solving methods based on configuration.

        Args:
            x_univl: Set of variables to universally quantify

        Returns:
            PySMT model if satisfiable, None otherwise
        """
        if self.qsolve == 'std':
            return get_model(ForAll(x_univl, self.formula), solver_name=self.sname)
        elif self.qsolve == 'z3qe':
            # Use quantifier elimination
            formula = qelim(ForAll(x_univl, self.formula))
            return get_model(formula, solver_name=self.sname)
        else:
            # Use CEGAR-based quantifier solving
            return get_qmodel(
                x_univl,
                self.formula,
                solver_name=self.sname,
                verbose=(self.verb > 2),
            )
