"""
Minimal Satisfying Assignment. adapted from algorithm by Alessandro Previti, Alexey S. Ignatiev
"""

from typing import FrozenSet

import z3

# from z3.z3util import get_vars
from aria.utils.z3.expr import get_variables


class MSASolver:
    """Minimal Satisfying Assignment solver using Z3."""

    def __init__(self, verbose=1):
        """Initialize the MSA solver."""

        self.formula = None
        # self.formula = simplify(self.formula)
        self.fvars = None
        self.verb = verbose

    def init_from_file(self, filename: str) -> None:
        """Initialize solver from an SMT2 file."""
        self.formula = z3.And(z3.parse_smt2_file(filename))
        # self.formula = simplify(self.formula)
        self.fvars = frozenset(get_variables(self.formula))

        if self.verb > 2:
            print(f"c formula: '{self.formula}'")

    def init_from_formula(self, formula: z3.BoolRef) -> None:
        """Initialize solver from a Z3 formula."""
        self.formula = formula
        # self.formula = simplify(self.formula)
        self.fvars = frozenset(get_variables(self.formula))

        if self.verb > 2:
            print(f"c formula: '{self.formula}'")

    def validate_small_model(self, model: z3.ModelRef) -> bool:
        """Check whether a small model is a 'sufficient condition'"""
        decls = model.decls()
        model_cnts = []
        for var in get_variables(self.formula):
            if var.decl() in decls:
                model_cnts.append(var == model[var])
        # print(model_cnts)
        # check entailment
        s = z3.Solver()
        s.add(z3.Not(z3.Implies(z3.And(model_cnts), self.formula)))
        if s.check() == z3.sat:
            return False
        return True

    def find_small_model(self):
        """This method implements find_msa() procedure from Fig. 2
        of the dillig-cav12 paper.
        """
        # testing if formula is satisfiable
        s = z3.Solver()
        s.add(self.formula)
        if s.check() == z3.unsat:
            return False

        mus = self.compute_mus(frozenset([]), self.fvars, 0)

        model = self.get_model_forall(mus)
        return model
        # return ['{0}={1}'.format(v, model[v]) for v in frozenset(self.fvars) - mus]

    def compute_mus(
        self, x_set: FrozenSet, fvars: FrozenSet, lb: int
    ):  # pylint: disable=invalid-name
        """
        Algorithm implements find_mus() procedure from Fig. 1
        of the dillig-cav12 paper.
        x_set: variables to include (named X in the paper)
        """
        if not fvars or len(fvars) <= lb:
            return frozenset()

        best = set()
        x_var = frozenset([next(iter(fvars))])  # should choose x in a more clever way

        if self.verb > 1:
            print(f"c state: X = {list(x_set)} + {list(x_var)}, lb = {lb}")

        if self.get_model_forall(x_set.union(x_var)):
            y_set = self.compute_mus(x_set.union(x_var), fvars - x_var, lb - 1)

            cost_curr = len(y_set) + 1
            if cost_curr > lb:
                best = y_set.union(x_var)
                lb = cost_curr

        y_set = self.compute_mus(x_set, frozenset(fvars) - x_var, lb)
        if len(y_set) > lb:
            best = y_set

        return best

    def get_model_forall(self, x_univl):
        """
        Get a model for the formula with the given variables universally quantified.

        Args:
            x_univl: List of variables to be universally quantified

        Returns:
            Model for the formula, or False if no model is found
        """
        s = z3.Solver()
        if len(x_univl) >= 1:
            qfml = z3.ForAll(list(x_univl), self.formula)
        else:
            qfml = self.formula
        s.add(qfml)
        if s.check() == z3.sat:
            return s.model()
        return False


if __name__ == "__main__":
    a, b, c, d = z3.Ints("a b c d")
    fml = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))
    ms = MSASolver()
    ms.init_from_formula(fml)
    print(ms.find_small_model())  # a = 3, b = 3
    # qfml = ForAll([c, d], fml)
    # s = Solver()
    # s.add(qfml)
    # s.check()
    # print(s.model())  # [a = 3, b = 3]
