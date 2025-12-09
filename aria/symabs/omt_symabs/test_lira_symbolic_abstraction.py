# coding: utf-8
import z3
from aria.tests.formula_generator import FormulaGenerator
# from aria.tests.grammar_gene import generate_from_grammar_as_str
from aria.symabs.omt_symabs.lira_symbolic_abstraction import LIRASymbolicAbstraction, OMTEngineType


def is_sat(e):
    s = z3.Solver()
    s.add(e)
    s.set("timeout", 6000)
    return s.check() == z3.sat


def test_lira_symbolic_abstraction():
    try:
        w, x, y, z = z3.Ints("w x y z")
        fg = FormulaGenerator([w, x, y, z])
        fml = fg.generate_formula()
        if is_sat(fml):
            # print(fml)
            # 1. Do abstraction
            sa = LIRASymbolicAbstraction()
            sa.init_from_fml(fml)

            sa.omt_engine.compact_opt = False
            sa.set_omt_engine_type(OMTEngineType.OptiMathSAT)
            # sa.set_omt_engine_type(OMTEngineType.Z3OPT)
            sa.interval_abs()
            # sa.zone_abs()
            print("Finish interval via OptiMathSAT")
            print("Res: \n", sa.interval_abs_as_fml)

            sa.set_omt_engine_type(OMTEngineType.Z3OPT)
            sa.interval_abs()
            print("Finish interval via Z3")
            print("Res: \n", sa.interval_abs_as_fml)

            # print("Finish zone")
            # sa.octagon_abs()
            # print("Finish octagon")

            # print("Zone: \n", sa.zone_abs_as_fml)
            # print("Octagon: \n", sa.octagon_abs_as_fml)
            # print(sa.octagon_abs_as_fml)
            return True

        return False
    except Exception as ex:
        print(ex)
        return False


if __name__ == '__main__':
    for _ in range(3):
        if test_lira_symbolic_abstraction():
            break
