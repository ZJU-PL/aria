"""
Minimal Satisfying Assignment (MSA) algorithm implementation.

This module provides an implementation of the Minimal Satisfying Assignment algorithm,
adapted from the algorithm by Alessandro Previti and Alexey S. Ignatiev. It contains
the MSASolver class which is used to find the minimal satisfying assignment for a given formula.

Note:
    MSA finding is a special case of optimization modulo theory.
"""
import logging
from typing import FrozenSet, Optional

import z3

from arlib.utils.z3_expr_utils import get_expr_vars

logger = logging.getLogger(__name__)


class MSASolver:
    """Mistral solver for finding Minimal Satisfying Assignments."""

    def __init__(self, verbose: int = 1) -> None:
        """Initialize the MSA solver.

        Args:
            verbose: Verbosity level (0=silent, 1=normal, 2+=debug)
        """
        self.formula: Optional[z3.BoolRef] = None
        self.fvars: Optional[FrozenSet[z3.ExprRef]] = None
        self.verb = verbose

    def init_from_file(self, filename: str) -> None:
        """Initialize solver from an SMT2 file.

        Args:
            filename: Path to the SMT2 file containing the formula
        """
        self.formula = z3.And(z3.parse_smt2_file(filename))
        self.fvars = frozenset(get_expr_vars(self.formula))

        if self.verb > 2:
            logger.debug("Formula: %s", self.formula)

    def init_from_formula(self, formula: z3.BoolRef) -> None:
        """Initialize solver from a Z3 formula.

        Args:
            formula: Z3 boolean formula to solve
        """
        self.formula = formula
        self.fvars = frozenset(get_expr_vars(self.formula))

        if self.verb > 2:
            logger.debug("Formula: %s", self.formula)

    def validate_small_model(self, model: z3.ModelRef) -> bool:
        """Check whether a small model is a sufficient condition.

        Validates that the model constraints entail the formula.

        Args:
            model: Z3 model to validate

        Returns:
            True if the model is a sufficient condition, False otherwise
        """
        decls = model.decls()
        model_constraints = []
        for var in get_expr_vars(self.formula):
            if var.decl() in decls:
                model_constraints.append(var == model[var])

        # Check entailment: model_constraints => formula
        solver = z3.Solver()
        solver.add(z3.Not(z3.Implies(z3.And(model_constraints), self.formula)))
        return solver.check() != z3.sat

    def find_small_model(self) -> Optional[z3.ModelRef]:
        """Find a minimal satisfying assignment.

        Implements the find_msa() procedure from Fig. 2 of the dillig-cav12 paper.

        Returns:
            Z3 model if satisfiable, False if unsatisfiable, None on error
        """
        # Test if formula is satisfiable
        solver = z3.Solver()
        solver.add(self.formula)
        if solver.check() == z3.unsat:
            return None

        mus = self.compute_mus(frozenset([]), self.fvars, 0)
        model = self.get_model_forall(mus)
        return model

    def compute_mus(
        self,
        X: FrozenSet[z3.ExprRef],
        fvars: FrozenSet[z3.ExprRef],
        lb: int,
    ) -> FrozenSet[z3.ExprRef]:
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

        best: FrozenSet[z3.ExprRef] = frozenset()
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

        Y = self.compute_mus(X, frozenset(fvars) - x, lb)
        if len(Y) > lb:
            best = Y

        return best

    def get_model_forall(
        self, x_univl: FrozenSet[z3.ExprRef]
    ) -> Optional[z3.ModelRef]:
        """Get a model for a universally quantified formula.

        Args:
            x_univl: Set of variables to universally quantify

        Returns:
            Z3 model if satisfiable, None otherwise
        """
        solver = z3.Solver()
        if x_univl:
            qfml = z3.ForAll(list(x_univl), self.formula)
        else:
            # No universal quantification needed
            qfml = self.formula
        solver.add(qfml)
        if solver.check() == z3.sat:
            return solver.model()
        return None


if __name__ == "__main__":
    """Demo: Find minimal satisfying assignment."""
    a, b, c, d = z3.Ints('a b c d')
    fml = z3.Or(
        z3.And(a == 3, b == 3),
        z3.And(a == 1, b == 1, c == 1, d == 1)
    )
    ms = MSASolver()
    ms.init_from_formula(fml)
    result = ms.find_small_model()
    logger.info("Result: %s", result)  # Expected: a = 3, b = 3
