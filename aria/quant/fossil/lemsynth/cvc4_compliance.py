# Module containing some utlities for converting between difference in syntax between z3 and cvc4.

import z3
from aria.quant.fossil.naturalproofs.utils import transform_expression
import aria.quant.fossil.lemsynth.options as options


def cvc4_compliant_formula_sexpr(formula):
    # Replace z3's select with cvc4's member and z3's store with cvc4's insert
    # Order of arguments is reverse for member and insert
    member = z3.Function('member', z3.IntSort(), z3.SetSort(z3.IntSort()), z3.BoolSort())
    insert = z3.Function('insert', z3.IntSort(), z3.SetSort(z3.IntSort()), z3.SetSort(z3.IntSort()))
    cond_mem = lambda expr: z3.is_select(expr)
    cond_ins = lambda expr: z3.is_store(expr)
    op_mem = lambda expr: member(expr.arg(1), expr.arg(0))
    op_ins = lambda expr: insert(expr.arg(1), expr.arg(0))
    trans_mem = transform_expression(formula, [(cond_mem, op_mem)])
    trans_ins = transform_expression(trans_mem, [(cond_ins, op_ins)])
    formula_sexpr = trans_ins.sexpr()
    # Replace occurrences of string corresponding to the emptyset in z3 with the one in cvc4
    z3_emptyset_str = '((as const (Array Int Bool)) false)'
    cvc4_emptyset_str = '(as emptyset (Set Int) )'
    z3_emptyset_str_new = 'empIntSet'
    if options.synthesis_solver == options.minisy:
        new_formula_sexpr = formula_sexpr.replace(z3_emptyset_str, z3_emptyset_str_new)
    else:
        new_formula_sexpr = formula_sexpr.replace(z3_emptyset_str, cvc4_emptyset_str)
    return new_formula_sexpr
