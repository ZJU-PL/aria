"""
Using pySMT to compute propositional itp
"""

import z3
from pysmt.shortcuts import binary_interpolant, sequence_interpolant
from pysmt.shortcuts import Solver, Interpolator
from pysmt.shortcuts import Symbol
from pysmt.typing import BOOL

from aria.utils.z3_expr_utils import get_variables


def to_pysmt_fml(fml: z3.ExprRef):
    # the following two lines are just for "fixing" some warnings
    # zvs = z3.z3util.get_vars(fml) # can be slow
    zvs = get_variables(fml)
    pysmt_vars = [Symbol(v.decl().name(), BOOL) for v in zvs]
    z3s = Solver(name='z3')
    pysmt_fml = z3s.converter.back(fml)
    return pysmt_vars, pysmt_fml


def pysmt_binary_itp(fml_a: z3.ExprRef, fml_b: z3.ExprRef) -> z3.ExprRef:
    """ Use pysmt to compute the binary interpolant and return a z3 expr"""
    _, pysmt_fml_a = to_pysmt_fml(fml_a)
    _, pysmt_fml_b = to_pysmt_fml(fml_b)

    itp = Interpolator(name="msat")
    res = itp.binary_interpolant(pysmt_fml_a, pysmt_fml_b)
    return Solver(name='z3').converter.convert(res)


"""
def pysmt_sequence_itp(formulas: [z3.ExprRef]):
    pysmt_formulas = []
    for fml in formulas:
        _, pysmt_fml_a = to_pysmt_fml(fml)
        pysmt_formulas.append(pysmt_fml_a)

    itp = sequence_interpolant(pysmt_formulas)
    return itp
"""

def demo_pysmt_itp():
    x, y, z = z3.Bools("x y z")
    fml_a = z3.And(x, y)
    fml_b = z3.And(z3.Not(x), z3.Not(y), z)
    itp = pysmt_binary_itp(fml_a, fml_b)
    if isinstance(itp, z3.ExprRef):
        print("success!")


if __name__ == '__main__':
    demo_pysmt_itp()
