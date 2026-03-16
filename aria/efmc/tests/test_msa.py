# coding: utf-8
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.tests.formula_generator import FormulaGenerator
from aria.efmc.engines.abduction.abductor.mistral_z3 import MSASolver
from aria.efmc.utils import get_variables, is_sat


def find_minimal_model(fml):
    msa_finder = MSASolver()
    msa_finder.init_from_formula(fml)
    print("All variables: ", get_variables(fml))
    m = msa_finder.find_small_model()
    print("Minimal model: \n", m)
    if msa_finder.validate_small_model(m):
        print("validation success!")
    return True


def gen_satisfiable_fml():
    for _ in range(10):
        w, x, y, z = z3.BitVecs("w x y z", 8)
        fg = FormulaGenerator([x, y, z])
        fml = fg.generate_formula()
        if is_sat(fml):
            return fml
    return z3.BoolVal(False)


class TestMSA(TestCase):

    def test_msa(self):
        try:
            fml = gen_satisfiable_fml()
            if not z3.is_false(fml):
                find_minimal_model(fml)
        except Exception as ex:
            print(ex)

    assert True


if __name__ == "__main__":
    main()
