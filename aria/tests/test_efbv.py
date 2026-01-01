# coding: utf-8
"""
For testing CDCL(T)-based parallel SMT solving engine
"""

import z3

from aria.tests import TestCase, main
from aria.tests.formula_generator import FormulaGenerator
from aria.quant.efbv.efbv_parallel.efbv_cegis_parallel import ParallelEFBVSolver
from aria.quant.efbv.efbv_parallel.efbv_utils import EFBVResult
from aria.utils.z3_expr_utils import get_variables


def gen_small_bv_formula(logic: str):
    assert logic == "bv"
    w, x, y, z = z3.BitVecs("w x y z", 3)
    fg = FormulaGenerator([w, x, y, z])
    fml = fg.generate_formula()
    existential_vars = [w, x, y]
    universal_vars = [z]
    return existential_vars, universal_vars, fml


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then("simplify", "solve-eqs")(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


def solve_with_z3(universal_vars, fml):
    sol = z3.SolverFor("BV")
    sol.add(z3.ForAll(universal_vars, fml))
    # set timeout
    sol.set("timeout", 10000)
    res = sol.check()
    if res == z3.sat:
        return EFBVResult.SAT, sol.model()
    elif res == z3.unsat:
        return EFBVResult.UNSAT, None
    else:
        return EFBVResult.UNKNOWN, None


class TestEFBVSolver(TestCase):

    def test_efbv_solver(self):
        for _ in range(15):
            existential_vars, universal_vars, fml = gen_small_bv_formula("bv")
            vars_fml = [str(v) for v in get_variables(fml)]
            if not ("w" in vars_fml and "x" in vars_fml and "y" in vars_fml):
                continue
            if is_simple_formula(fml):
                continue
            # print(fml)
            # print(existential_vars)
            # seq_solver = SequentialEFBVSolver(mode="z3")
            # res_seq = seq_solver.solve_efsmt_bv(existential_vars, universal_vars, fml)
            par_solver = ParallelEFBVSolver(mode="canary")
            res_par = par_solver.solve_efsmt_bv(existential_vars, universal_vars, fml)

            res_b, model = solve_with_z3(universal_vars, fml)
            if res_par != res_b:
                print("inconsistent!!")
                print(res_par, res_b)
                print(fml)
                print(model)
                break
            print(res_par, res_b)
            # break


if __name__ == "__main__":
    main()
