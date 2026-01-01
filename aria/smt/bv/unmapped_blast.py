"""
Converting QF_BV to SAT without tracking mappings between BV variables
and SAT variables.
"""

import z3
from pysat.formula import CNF
from pysat.solvers import Solver
from aria.utils.types import SolverResult
from aria.utils.z3_expr_utils import get_variables

qfbv_tactic = z3.AndThen(
    z3.With("simplify", flat_and_or=False),
    z3.Tactic("elim-uncnstr"),
    z3.Tactic("reduce-bv-size"),
    z3.With("simplify", som=True, pull_cheap_ite=True, flat=True),
    z3.Tactic("max-bv-sharing"),
    z3.Tactic("bit-blast"),
    z3.Tactic("tseitin-cnf"),
)


def qfbv_to_sat(fml: z3.ExprRef, solver_name: str = "minisat22") -> SolverResult:
    """Convert a QF_BV formula to SAT and solve it."""
    if not isinstance(fml, z3.ExprRef):
        raise ValueError("Input must be a Z3 expression")

    if not all(z3.is_bv(arg) for arg in get_variables(fml)):
        raise ValueError("Formula must contain only bit-vector operations")

    try:
        after_simp = qfbv_tactic(fml).as_expr()

        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        if z3.is_true(after_simp):
            return SolverResult.SAT

        g = z3.Goal()
        g.add(after_simp)
        pos = CNF(from_string=g.dimacs())

        with Solver(name=solver_name, bootstrap_with=pos) as aux:
            return SolverResult.SAT if aux.solve() else SolverResult.UNSAT

    except Exception as e:
        raise RuntimeError(f"Conversion/solving failed: {str(e)}") from e


def demo() -> None:
    """Demonstrate QF_BV to SAT conversion."""
    x, y = z3.BitVecs("x y", 6)
    fml1 = z3.And(x + y == 8, x - y == 2)
    print("Example 1:", qfbv_to_sat(fml1))

    a, b = z3.BitVecs("a b", 4)
    fml2 = z3.And(z3.ULT(a, b), a & b == 0)
    print("Example 2:", qfbv_to_sat(fml2))


if __name__ == "__main__":
    demo()
