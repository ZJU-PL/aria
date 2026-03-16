"""
Utilities for Z3 uninterpreted functions (UF).
"""

from __future__ import print_function
from typing import Callable, Optional, Dict
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
    """Purify formula: introduce fresh vars for mixed-theory terms (no quantifiers)."""

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
                    l, r = expr.arg(0), expr.arg(1)
                    if (is_arith(l) and is_uf_term(r)) or (
                        is_uf_term(l) and is_arith(r)
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


def test_replace():  # pylint: disable=too-many-locals
    """Test function for replace_func_with_template."""
    x, y = z3.Ints("x y")
    fml = x + x + y > 2
    seen = {}
    for exp in visitor(fml, seen):
        if z3.is_const(exp) and exp.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            print("Variable", exp)
        else:
            print(exp)
    s = z3.SolverFor("HORN")
    inv = z3.Function("inv", z3.IntSort(), z3.IntSort(), z3.BoolSort())
    i, ip, j, jp = z3.Ints("i ip j jp")
    s.add(z3.ForAll([i, j], z3.Implies(i == 0, inv(i, j))))
    s.add(
        z3.ForAll(
            [i, ip, j, jp],
            z3.Implies(z3.And(inv(i, j), i < 10, ip == i + 1), inv(ip, jp)),
        )
    )
    s.add(z3.ForAll([i, j], z3.Implies(z3.And(inv(i, j), i >= 10), i == 10)))
    a0, a1, a2 = z3.Ints("a0 a1 a2")
    b0, b1, b2 = z3.Ints("b0 b1 b2")
    x = z3.Var(0, z3.IntSort())
    y = z3.Var(1, z3.IntSort())
    template = z3.And(a0 + a1 * x + a2 * y >= 0, b0 + b1 * x + b2 * y >= 0)
    chc = z3.And(s.assertions())
    print(replace_func_with_template(chc, inv, template))


def test_instiatiate():
    """Test function for instiatiate_func_with_axioms."""
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort(), z3.BoolSort())
    axiom = z3.ForAll([x, y], z3.Implies(f(x, y), x + y > 0))
    a, b, c = z3.Ints("a b c")
    formula = z3.And(
        f(x, y),
        f(y, x),
        z3.Implies(f(a, b), f(b, c)),
        z3.Or(f(x + 1, y - 1), f(x * 2, y / 2)),
        z3.Not(f(0, 0)),
    )
    print(instiatiate_func_with_axioms(formula, f, axiom))


def test_purify():
    """Test function for purify."""
    x, y, z = z3.Ints("x y z")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    mixed_formula = z3.And(f(x + y) > z, f(x) + f(y) == 10, z > 0)
    print("Original:", mixed_formula)
    purified = purify(mixed_formula)
    print("Purified:", purified)
    s1, s2 = z3.Solver(), z3.Solver()
    s1.add(mixed_formula)
    s2.add(purified)
    print("Original sat:", s1.check())
    print("Purified sat:", s2.check())
    if s1.check() == s2.check() == z3.sat:
        print("Original model:", s1.model())
        print("Purified model:", s2.model())
    print("--- Quantifier test ---")
    a, b = z3.Ints("a b")
    g = z3.Function("g", z3.IntSort(), z3.IntSort(), z3.IntSort())
    quantified_formula = z3.ForAll([a, b], z3.Implies(a > b, g(a + b, a - b) > g(a, b)))
    print("Quantified:", quantified_formula)
    try:
        purified_quantified = purify(quantified_formula)
        print("Purified quantified:", purified_quantified)
    except ValueError as e:
        print("Error:", e)
        print("Quantified formulas are rejected.")


if __name__ == "__main__":
    test_replace()
    print("-" * 40)
    test_instiatiate()
    print("-" * 40)
    test_purify()
