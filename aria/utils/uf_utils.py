"""Utilities for Z3 uninterpreted functions."""

from __future__ import print_function

from typing import Callable, Dict, Optional

import z3


def visitor(exp: z3.ExprRef, seen: Dict[z3.ExprRef, bool]):
    """Yield all subexpressions in a Z3 expression."""
    if exp in seen:
        return
    seen[exp] = True
    yield exp
    if z3.is_app(exp):
        for ch in exp.children():
            yield from visitor(ch, seen)
    elif z3.is_quantifier(exp):
        yield from visitor(exp.body(), seen)


def modify(
    expression: z3.ExprRef, fn: Callable[[z3.ExprRef], Optional[z3.ExprRef]]
) -> z3.ExprRef:
    """Apply fn to each subexpression of a Z3 expression."""
    seen = {}

    def visit(exp):
        if exp in seen:
            pass
        elif fn(exp) is not None:
            seen[exp] = fn(exp)
        elif z3.is_and(exp):
            seen[exp] = z3.And([visit(ch) for ch in exp.children()])
        elif z3.is_or(exp):
            seen[exp] = z3.Or([visit(ch) for ch in exp.children()])
        elif z3.is_app(exp):
            seen[exp] = exp.decl()([visit(ch) for ch in exp.children()])
        elif z3.is_quantifier(exp):
            body = visit(exp.body())
            is_forall = exp.is_forall()
            num_pats = exp.num_patterns()
            pats = (z3.Pattern * num_pats)()
            for i in range(num_pats):
                pats[i] = exp.pattern(i).ast
            num_decls = exp.num_vars()
            sorts = (z3.Sort * num_decls)()
            names = (z3.Symbol * num_decls)()
            for i in range(num_decls):
                sorts[i] = exp.var_sort(i).ast
                names[i] = z3.to_symbol(exp.var_name(i), exp.ctx)
            r = z3.QuantifierRef(
                z3.Z3_mk_quantifier(
                    exp.ctx_ref(),
                    is_forall,
                    exp.weight(),
                    num_pats,
                    pats,
                    num_decls,
                    sorts,
                    names,
                    body.ast,
                ),
                exp.ctx,
            )
            seen[exp] = r
        else:
            seen[exp] = exp
        return seen[exp]

    return visit(expression)


def replace_func_with_template(
    formula: z3.ExprRef, func: z3.FuncDeclRef, template: z3.ExprRef
) -> z3.ExprRef:
    """Replace UF func in formula with template."""

    def update(expression):
        if z3.is_app(expression) and z3.eq(expression.decl(), func):
            args = [expression.arg(i) for i in range(expression.num_args())]
            return z3.substitute_vars(template, *args)
        return None

    return modify(formula, update)


def instiatiate_func_with_axioms(
    formula: z3.ExprRef, func: z3.FuncDeclRef, axiom: z3.ExprRef
) -> z3.ExprRef:
    """Instantiate UF func in formula with axiom."""
    func_apps = []
    seen = {}
    for expr in visitor(formula, seen):
        if z3.is_app(expr) and z3.eq(expr.decl(), func):
            func_apps.append(expr)
    if not func_apps:
        return formula
    if not z3.is_quantifier(axiom):
        raise ValueError("Axiom must be a quantified formula")
    axiom_body = axiom.body()
    instantiated_axioms = []
    for app in func_apps:
        args = [app.arg(i) for i in range(app.num_args())]
        num_vars = axiom.num_vars()
        if num_vars != len(args):
            raise ValueError(f"#args ({len(args)}) != #quantified vars ({num_vars})")
        instantiated = z3.substitute_vars(axiom_body, *args)
        instantiated_axioms.append(instantiated)
    return z3.And(formula, *instantiated_axioms) if instantiated_axioms else formula


def purify(formula: z3.ExprRef) -> z3.ExprRef:
    """Purify formula: introduce fresh vars for mixed-theory terms."""

    def contains_quantifier(expr, seen=None):
        if seen is None:
            seen = {}
        if expr in seen:
            return seen[expr]
        if z3.is_quantifier(expr):
            seen[expr] = True
            return True
        if z3.is_app(expr):
            for i in range(expr.num_args()):
                if contains_quantifier(expr.arg(i), seen):
                    seen[expr] = True
                    return True
        seen[expr] = False
        return False

    if contains_quantifier(formula):
        raise ValueError("Quantified formulas not supported")
    processed, fresh_vars, equalities = {}, {}, []

    def is_uf_term(expr):
        return z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED

    def is_arith(expr):
        if z3.is_int(expr) or z3.is_real(expr):
            return True
        if z3.is_const(expr) and (
            expr.sort() == z3.IntSort() or expr.sort() == z3.RealSort()
        ):
            return True
        if z3.is_app(expr):
            return expr.decl().kind() in [
                z3.Z3_OP_ADD,
                z3.Z3_OP_SUB,
                z3.Z3_OP_MUL,
                z3.Z3_OP_DIV,
                z3.Z3_OP_IDIV,
                z3.Z3_OP_MOD,
                z3.Z3_OP_REM,
                z3.Z3_OP_POWER,
                z3.Z3_OP_LT,
                z3.Z3_OP_LE,
                z3.Z3_OP_GT,
                z3.Z3_OP_GE,
            ]
        return False

    def is_mixed(expr):
        if z3.is_const(expr) or z3.is_var(expr):
            return False
        if z3.is_app(expr):
            if expr.decl().kind() in [
                z3.Z3_OP_EQ,
                z3.Z3_OP_LT,
                z3.Z3_OP_LE,
                z3.Z3_OP_GT,
                z3.Z3_OP_GE,
            ]:
                if expr.num_args() == 2:
                    left, right = expr.arg(0), expr.arg(1)
                    if (is_arith(left) and is_uf_term(right)) or (
                        is_uf_term(left) and is_arith(right)
                    ):
                        return True
            if is_arith(expr):
                return any(
                    is_uf_term(expr.arg(i)) or is_mixed(expr.arg(i))
                    for i in range(expr.num_args())
                )
            if is_uf_term(expr):
                return any(
                    is_arith(expr.arg(i)) or is_mixed(expr.arg(i))
                    for i in range(expr.num_args())
                )
            return any(is_mixed(expr.arg(i)) for i in range(expr.num_args()))
        return False

    def purify_term(expr):
        if expr in processed:
            return processed[expr]
        if z3.is_const(expr) or z3.is_var(expr):
            processed[expr] = expr
            return expr
        if is_mixed(expr):
            if expr in fresh_vars:
                return fresh_vars[expr]
            fresh_var = z3.Const(f"purify_{len(fresh_vars)}", expr.sort())
            fresh_vars[expr] = fresh_var
            purified_expr = expr
            if z3.is_app(expr):
                args = [purify_term(expr.arg(i)) for i in range(expr.num_args())]
                if z3.is_and(expr):
                    purified_expr = z3.And(args)
                elif z3.is_or(expr):
                    purified_expr = z3.Or(args)
                else:
                    purified_expr = expr.decl()(*args)
            equalities.append(fresh_var == purified_expr)
            processed[expr] = fresh_var
            return fresh_var
        if z3.is_app(expr):
            args = [purify_term(expr.arg(i)) for i in range(expr.num_args())]
            result = expr.decl()(*args)
            processed[expr] = result
            return result
        processed[expr] = expr
        return expr

    purified_formula = purify_term(formula)
    return z3.And(purified_formula, *equalities) if equalities else purified_formula
