"""
Minimal Satisfying Assignment. adapted from algorithm by Alessandro Previti, Alexey S. Ignatiev
"""

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Bool, get_model, Not, Solver, qelim, ForAll
from pysmt.smtlib.parser import SmtLibParser


# ==============================================================================
def get_qmodel(x_univl, formula, maxiters=None, solver_name=None, verbose=False):
    """
    A simple 2QBF CEGAR implementation for SMT.
    """

    x_univl = set(x_univl)
    x_exist = formula.get_free_variables() - x_univl

    with Solver(name=solver_name) as asolver:
        asolver.add_assertion(Bool(True))
        iters = 0

        while maxiters is None or iters <= maxiters:
            iters += 1

            amodel = asolver.solve()
            if not amodel:
                return None
            cand = {v: asolver.get_value(v) for v in x_exist}
            subform = formula.substitute(cand).simplify()
            if verbose:
                print(f"c qsolve cand{iters}: {cand}")

            cmodel = get_model(Not(subform), solver_name=solver_name)
            if cmodel is None:
                return cand
            coex = {v: cmodel[v] for v in x_univl}
            subform = formula.substitute(coex).simplify()
            if verbose:
                print(f"c qsolve coex{iters}: {coex}")

            asolver.add_assertion(subform)

        raise SolverReturnedUnknownResultError


#
# ==============================================================================
class Mistral:
    """
    Mistral solver class.
    """

    def __init__(
        self, simplify, solver, qsolve, verbose, fname
    ):  # pylint: disable=too-many-arguments
        """
        Constructor.
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
            print(f"c formula: '{self.formula}'")

        if self.verb > 1:
            print(f"c vars ({len(self.fvars)}):", list(self.fvars))

    def solve(self):
        """
        This method implements find_msa() procedure from Fig. 2
        of the dillig-cav12 paper.
        """

        # testing if formula is satisfiable
        if not get_model(self.formula, solver_name=self.sname):
            return None

        mus = self.compute_mus(frozenset([]), self.fvars, 0)

        model = self.get_model_forall(mus)
        return [f"{v}={model[v]}" for v in self.fvars - mus]

    def compute_mus(self, x_set, fvars, lb):  # pylint: disable=invalid-name
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

        y_set = self.compute_mus(x_set, fvars - x_var, lb)
        if len(y_set) > lb:
            best = y_set

        return best

    def get_model_forall(self, x_univl):
        """
        Calls either pysmt.shortcuts.get_model() or get_qmodel().
        """

        if self.qsolve == "std":
            return get_model(ForAll(x_univl, self.formula), solver_name=self.sname)
        elif self.qsolve == "z3qe":
            formula = qelim(ForAll(x_univl, self.formula))
            return get_model(formula, solver_name=self.sname)
        return get_qmodel(
            x_univl, self.formula, solver_name=self.sname, verbose=self.verb > 2
        )
