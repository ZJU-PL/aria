import z3
from aria.tests.formula_generator import FormulaGenerator
from aria.tests.grammar_gene import gene_smt2string
from aria.utils import SolverResult


def gen_small_formula(logic: str):
    if logic == "int":
        w, x, y, z = z3.Ints("w x y z")
    elif logic == "real":
        w, x, y, z = z3.Reals("w x y z")
    elif logic == "bv":
        w, x, y, z = z3.BitVecs("w x y z", 8)
    else:
        w, x, y, z = z3.Reals("w x y z")
    fg = FormulaGenerator([x, y, z])
    fml = fg.generate_formula()
    s = z3.Solver()
    s.add(fml)
    return s.to_smt2()


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then("simplify", "elim-uncnstr", "solve-eqs", "tseitin-cnf")(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


def solve_with_z3(smt2string: str):
    fml = z3.And(z3.parse_smt2_string(smt2string))
    sol = z3.Solver()
    sol.add(fml)
    res = sol.check()
    if res == z3.sat:
        return SolverResult.SAT
    elif res == z3.unsat:
        return SolverResult.UNSAT
    else:
        return SolverResult.UNKNOWN
