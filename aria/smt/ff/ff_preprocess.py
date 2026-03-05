#!/usr/bin/env python3
"""
ff_preprocess.py  –  Lightweight normalization and gadget rewrites for QF_FF.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .ff_ast import (
    BoolAnd,
    BoolConst,
    BoolIte,
    BoolImplies,
    BoolNot,
    BoolOr,
    BoolVar,
    BoolXor,
    FieldAdd,
    FieldConst,
    FieldDiv,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldSub,
    FieldVar,
    ParsedFormula,
    infer_field_modulus,
)


def preprocess_formula(formula: ParsedFormula) -> ParsedFormula:
    """Normalize field arithmetic and add safe gadget implications."""
    rewritten_assertions: List[FieldExpr] = []
    for assertion in formula.assertions:
        normalized = _rewrite(assertion, formula.variables)
        rewritten_assertions.extend(_split_top_level_and(normalized))

    derived_assertions = _derive_is_zero_implications(
        rewritten_assertions, formula.variables
    )
    all_assertions = rewritten_assertions + derived_assertions

    return ParsedFormula(
        formula.field_size,
        formula.variables,
        [_rewrite(assertion, formula.variables) for assertion in all_assertions],
        expected_status=formula.expected_status,
        field_sizes=formula.field_sizes,
    )


def preprocess_formula_with_metadata(
    formula: ParsedFormula,
) -> Tuple[ParsedFormula, Dict[str, int]]:
    """Normalize a formula and expose lightweight preprocessing metadata.

    Metadata fields:
        input_assertions: assertions in the original parsed formula.
        rewritten_assertions: assertions after normalization/splitting.
        split_assertions: number of additional assertions introduced by split.
        derived_is_zero_assertions: assertions generated from gadget inference.
        output_assertions: assertions in the final normalized formula.
    """
    rewritten_assertions: List[FieldExpr] = []
    split_assertions = 0
    for assertion in formula.assertions:
        normalized = _rewrite(assertion, formula.variables)
        split = _split_top_level_and(normalized)
        split_assertions += max(0, len(split) - 1)
        rewritten_assertions.extend(split)

    derived_assertions = _derive_is_zero_implications(
        rewritten_assertions, formula.variables
    )
    all_assertions = rewritten_assertions + derived_assertions
    normalized_formula = ParsedFormula(
        formula.field_size,
        formula.variables,
        [_rewrite(assertion, formula.variables) for assertion in all_assertions],
        expected_status=formula.expected_status,
        field_sizes=formula.field_sizes,
    )
    metadata = {
        "input_assertions": len(formula.assertions),
        "rewritten_assertions": len(rewritten_assertions),
        "split_assertions": split_assertions,
        "derived_is_zero_assertions": len(derived_assertions),
        "output_assertions": len(normalized_formula.assertions),
    }
    return normalized_formula, metadata


def _split_top_level_and(expr: FieldExpr) -> List[FieldExpr]:
    if isinstance(expr, BoolAnd):
        result: List[FieldExpr] = []
        for arg in expr.args:
            result.extend(_split_top_level_and(arg))
        return result
    return [expr]


def _rewrite(expr: FieldExpr, variables: Dict[str, str]) -> FieldExpr:
    if isinstance(expr, FieldConst) or isinstance(expr, FieldVar):
        return expr
    if isinstance(expr, BoolConst) or isinstance(expr, BoolVar):
        return expr

    if isinstance(expr, FieldAdd):
        return _rewrite_field_add(expr.args, variables)
    if isinstance(expr, FieldMul):
        return _rewrite_field_mul(expr.args, variables)
    if isinstance(expr, FieldSub):
        minuend = _rewrite(expr.args[0], variables)
        subtrahends = [_rewrite(arg, variables) for arg in expr.args[1:]]
        return _rewrite(
            FieldAdd(minuend, *[FieldNeg(arg) for arg in subtrahends]), variables
        )
    if isinstance(expr, FieldNeg):
        arg = _rewrite(expr.arg, variables)
        modulus = infer_field_modulus(arg, variables)
        if isinstance(arg, FieldConst) and modulus is not None:
            return FieldConst((-arg.value) % modulus, modulus)
        if isinstance(arg, FieldNeg):
            return arg.arg
        return FieldNeg(arg)
    if isinstance(expr, FieldPow):
        base = _rewrite(expr.base, variables)
        modulus = infer_field_modulus(base, variables)
        if expr.exponent == 0:
            return FieldConst(1, modulus)
        if expr.exponent == 1:
            return base
        if isinstance(base, FieldConst) and modulus is not None:
            return FieldConst(pow(base.value, expr.exponent, modulus), modulus)
        return FieldPow(base, expr.exponent)
    if isinstance(expr, FieldDiv):
        return FieldDiv(_rewrite(expr.num, variables), _rewrite(expr.denom, variables))

    if isinstance(expr, FieldEq):
        left = _rewrite(expr.left, variables)
        right = _rewrite(expr.right, variables)
        modulus = infer_field_modulus(FieldEq(left, right), variables)
        if modulus is not None:
            zero = FieldConst(0, modulus)
            if _is_zero_const(right, modulus):
                if isinstance(left, FieldConst):
                    return BoolConst(left.value == 0)
                rewritten = _rewrite_booleanity_constraint(left, modulus)
                if rewritten is not None:
                    return rewritten
                return FieldEq(left, zero)
            if _is_zero_const(left, modulus):
                rewritten = _rewrite_booleanity_constraint(right, modulus)
                if rewritten is not None:
                    return rewritten
                return FieldEq(right, zero)
            diff = _rewrite(FieldSub(left, right), variables)
            if isinstance(diff, FieldConst):
                return BoolConst(diff.value == 0)
            rewritten = _rewrite_booleanity_constraint(diff, modulus)
            if rewritten is not None:
                return rewritten
            return FieldEq(diff, zero)

        if isinstance(left, BoolConst) and isinstance(right, BoolConst):
            return BoolConst(left.value == right.value)
        return FieldEq(left, right)

    if isinstance(expr, BoolAnd):
        args = [_rewrite(arg, variables) for arg in expr.args]
        flat_args: List[FieldExpr] = []
        for arg in args:
            if isinstance(arg, BoolConst):
                if not arg.value:
                    return BoolConst(False)
                continue
            if isinstance(arg, BoolAnd):
                flat_args.extend(arg.args)
            else:
                flat_args.append(arg)
        if not flat_args:
            return BoolConst(True)
        if len(flat_args) == 1:
            return flat_args[0]
        return BoolAnd(*flat_args)

    if isinstance(expr, BoolOr):
        args = [_rewrite(arg, variables) for arg in expr.args]
        flat_args = []
        for arg in args:
            if isinstance(arg, BoolConst):
                if arg.value:
                    return BoolConst(True)
                continue
            if isinstance(arg, BoolOr):
                flat_args.extend(arg.args)
            else:
                flat_args.append(arg)
        if not flat_args:
            return BoolConst(False)
        if len(flat_args) == 1:
            return flat_args[0]
        return BoolOr(*flat_args)

    if isinstance(expr, BoolXor):
        args = [_rewrite(arg, variables) for arg in expr.args]
        flat_args = []
        parity = False
        for arg in args:
            if isinstance(arg, BoolConst):
                parity = parity ^ arg.value
            elif isinstance(arg, BoolXor):
                flat_args.extend(arg.args)
            else:
                flat_args.append(arg)
        if not flat_args:
            return BoolConst(parity)
        if len(flat_args) == 1:
            return _rewrite(BoolNot(flat_args[0]), variables) if parity else flat_args[0]
        core = BoolXor(*flat_args)
        return _rewrite(BoolNot(core), variables) if parity else core

    if isinstance(expr, BoolNot):
        arg = _rewrite(expr.arg, variables)
        if isinstance(arg, BoolConst):
            return BoolConst(not arg.value)
        if isinstance(arg, BoolNot):
            return arg.arg
        return BoolNot(arg)

    if isinstance(expr, BoolImplies):
        antecedent = _rewrite(expr.antecedent, variables)
        consequent = _rewrite(expr.consequent, variables)
        if isinstance(antecedent, BoolConst):
            return consequent if antecedent.value else BoolConst(True)
        if isinstance(consequent, BoolConst):
            return BoolConst(True) if consequent.value else _rewrite(
                BoolNot(antecedent), variables
            )
        return BoolImplies(antecedent, consequent)

    if isinstance(expr, BoolIte):
        cond = _rewrite(expr.cond, variables)
        then_expr = _rewrite(expr.then_expr, variables)
        else_expr = _rewrite(expr.else_expr, variables)
        if isinstance(cond, BoolConst):
            return then_expr if cond.value else else_expr
        return BoolIte(cond, then_expr, else_expr)

    return expr


def _rewrite_field_add(args: Sequence[FieldExpr], variables: Dict[str, str]) -> FieldExpr:
    rewritten = [_rewrite(arg, variables) for arg in args]
    flat_args: List[FieldExpr] = []
    for arg in rewritten:
        if isinstance(arg, FieldAdd):
            flat_args.extend(arg.args)
        else:
            flat_args.append(arg)

    modulus = None
    for arg in flat_args:
        modulus = infer_field_modulus(arg, variables)
        if modulus is not None:
            break
    if modulus is None:
        return FieldAdd(*flat_args)

    const_sum = 0
    others = []
    for arg in flat_args:
        if isinstance(arg, FieldConst):
            const_sum = (const_sum + arg.value) % modulus
        else:
            others.append(arg)
    if const_sum:
        others.append(FieldConst(const_sum, modulus))
    if not others:
        return FieldConst(0, modulus)
    if len(others) == 1:
        return others[0]
    return FieldAdd(*others)


def _rewrite_field_mul(args: Sequence[FieldExpr], variables: Dict[str, str]) -> FieldExpr:
    rewritten = [_rewrite(arg, variables) for arg in args]
    flat_args: List[FieldExpr] = []
    for arg in rewritten:
        if isinstance(arg, FieldMul):
            flat_args.extend(arg.args)
        else:
            flat_args.append(arg)

    modulus = None
    for arg in flat_args:
        modulus = infer_field_modulus(arg, variables)
        if modulus is not None:
            break
    if modulus is None:
        return FieldMul(*flat_args)

    const_prod = 1
    others = []
    for arg in flat_args:
        if isinstance(arg, FieldConst):
            const_prod = (const_prod * arg.value) % modulus
        else:
            others.append(arg)
    if const_prod == 0:
        return FieldConst(0, modulus)
    if const_prod != 1:
        others.insert(0, FieldConst(const_prod, modulus))
    if not others:
        return FieldConst(1, modulus)
    if len(others) == 1:
        return others[0]
    return FieldMul(*others)


def _rewrite_booleanity_constraint(
    expr: FieldExpr, modulus: int
) -> Optional[FieldExpr]:
    variable = _match_booleanity_product(expr, modulus)
    if variable is None:
        return None
    zero = FieldConst(0, modulus)
    one = FieldConst(1 % modulus, modulus)
    return BoolOr(FieldEq(variable, zero), FieldEq(variable, one))


def _match_booleanity_product(expr: FieldExpr, modulus: int) -> Optional[FieldExpr]:
    if not isinstance(expr, FieldMul):
        return None

    factors = list(expr.args)
    seen_var = None
    seen_var_minus_one = None
    for factor in factors:
        if isinstance(factor, FieldConst):
            if factor.value == 0:
                return None
            continue
        if isinstance(factor, FieldVar):
            seen_var = factor if seen_var is None else seen_var
            continue
        candidate = _match_var_minus_one(factor, modulus)
        if candidate is not None:
            seen_var_minus_one = candidate if seen_var_minus_one is None else seen_var_minus_one
            continue
        return None

    if seen_var is None or seen_var_minus_one is None:
        return None
    if _expr_key(seen_var) != _expr_key(seen_var_minus_one):
        return None
    return seen_var


def _match_var_minus_one(expr: FieldExpr, modulus: int) -> Optional[FieldExpr]:
    if not isinstance(expr, FieldAdd) or len(expr.args) != 2:
        return None
    left, right = expr.args
    minus_one = (modulus - 1) % modulus
    if isinstance(left, FieldConst) and left.value == minus_one:
        return right
    if isinstance(right, FieldConst) and right.value == minus_one:
        return left
    return None


def _derive_is_zero_implications(
    assertions: Sequence[FieldExpr], variables: Dict[str, str]
) -> List[FieldExpr]:
    del variables
    products = []
    links = []
    for assertion in assertions:
        product = _match_zero_product(assertion)
        if product is not None:
            products.append(product)
        link = _match_is_zero_link(assertion)
        if link is not None:
            links.append(link)

    derived: List[FieldExpr] = []
    seen: Set[Tuple[str, str, int]] = set()
    for z_name, x_name, modulus in products:
        key = tuple(sorted((z_name, x_name))) + (modulus,)
        if key in seen:
            continue
        for link_z, link_x, _m_name, link_modulus in links:
            if modulus != link_modulus:
                continue
            if link_z not in (z_name, x_name) or link_x not in (z_name, x_name):
                continue
            zero = FieldConst(0, modulus)
            one = FieldConst(1 % modulus, modulus)
            z_var = FieldVar(link_z)
            x_var = FieldVar(link_x)
            derived.append(BoolOr(FieldEq(z_var, zero), FieldEq(z_var, one)))
            derived.append(
                BoolAnd(
                    BoolImplies(FieldEq(z_var, one), FieldEq(x_var, zero)),
                    BoolImplies(FieldEq(x_var, zero), FieldEq(z_var, one)),
                )
            )
            seen.add(key)
            break
    return derived


def _match_zero_product(assertion: FieldExpr) -> Optional[Tuple[str, str, int]]:
    if not isinstance(assertion, FieldEq) or not isinstance(assertion.right, FieldConst):
        return None
    modulus = assertion.right.modulus
    if modulus is None or assertion.right.value != 0:
        return None
    lhs = assertion.left
    if not isinstance(lhs, FieldMul):
        return None
    vars_in_product = [factor for factor in lhs.args if isinstance(factor, FieldVar)]
    if len(vars_in_product) != 2:
        return None
    return (vars_in_product[0].name, vars_in_product[1].name, modulus)


def _match_is_zero_link(assertion: FieldExpr) -> Optional[Tuple[str, str, str, int]]:
    if not isinstance(assertion, FieldEq) or not isinstance(assertion.right, FieldConst):
        return None
    modulus = assertion.right.modulus
    if modulus is None or assertion.right.value != 0:
        return None
    lhs = assertion.left
    if not isinstance(lhs, FieldAdd):
        return None

    z_name = None
    m_name = None
    x_name = None
    const_value = 0
    for term in lhs.args:
        if isinstance(term, FieldConst):
            const_value = (const_value + term.value) % modulus
            continue
        if isinstance(term, FieldVar):
            if z_name is None:
                z_name = term.name
                continue
            return None
        if isinstance(term, FieldMul) and len(term.args) == 2:
            left, right = term.args
            if isinstance(left, FieldVar) and isinstance(right, FieldVar):
                m_name = left.name
                x_name = right.name
                continue
            if isinstance(left, FieldConst) and left.value == 1:
                continue
        return None

    if z_name is None or m_name is None or x_name is None:
        return None
    if const_value != (modulus - 1) % modulus:
        return None
    return (z_name, x_name, m_name, modulus)


def _is_zero_const(expr: FieldExpr, modulus: int) -> bool:
    return isinstance(expr, FieldConst) and expr.modulus == modulus and expr.value == 0


def _expr_key(expr: FieldExpr):
    if isinstance(expr, FieldVar):
        return ("FieldVar", expr.name)
    if isinstance(expr, FieldConst):
        return ("FieldConst", expr.value, expr.modulus)
    if isinstance(expr, FieldNeg):
        return ("FieldNeg", _expr_key(expr.arg))
    if isinstance(expr, FieldAdd) or isinstance(expr, FieldMul) or isinstance(expr, BoolAnd) or isinstance(expr, BoolOr) or isinstance(expr, BoolXor):
        return (type(expr).__name__, tuple(_expr_key(arg) for arg in expr.args))
    if isinstance(expr, FieldEq):
        return ("FieldEq", _expr_key(expr.left), _expr_key(expr.right))
    if isinstance(expr, BoolNot):
        return ("BoolNot", _expr_key(expr.arg))
    if isinstance(expr, BoolImplies):
        return ("BoolImplies", _expr_key(expr.antecedent), _expr_key(expr.consequent))
    if isinstance(expr, BoolIte):
        return (
            "BoolIte",
            _expr_key(expr.cond),
            _expr_key(expr.then_expr),
            _expr_key(expr.else_expr),
        )
    if isinstance(expr, BoolConst):
        return ("BoolConst", expr.value)
    return (type(expr).__name__, id(expr))
