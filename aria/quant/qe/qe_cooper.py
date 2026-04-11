"""Cooper-style existential QE for quantifier-free integer linear arithmetic."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple, cast

import z3


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def _normalize_vars(vars_or_var: Optional[Any]) -> List[z3.ExprRef]:
    if vars_or_var is None:
        return []
    if z3.is_expr(vars_or_var):
        return [cast(z3.ExprRef, vars_or_var)]
    return [cast(z3.ExprRef, var) for var in cast(Iterable[Any], vars_or_var)]


def _get_projection_vars(
    phi: z3.ExprRef,
    qvars: List[z3.ExprRef],
    keep_vars: Optional[List[z3.ExprRef]],
) -> List[z3.ExprRef]:
    if keep_vars is None:
        return qvars

    qvar_ids = {var.get_id() for var in qvars}
    keep_var_ids = {var.get_id() for var in keep_vars}
    if qvar_ids & keep_var_ids:
        raise ValueError("qvars and keep_vars must be disjoint")

    projection_vars = list(qvars)
    stack = [phi]
    seen = set()
    while stack:
        expr = stack.pop()
        expr_id = expr.get_id()
        if expr_id in seen:
            continue
        seen.add(expr_id)
        if z3.is_quantifier(expr):
            stack.append(cast(z3.QuantifierRef, expr).body())
            continue
        if z3.is_app(expr):
            if expr.num_args() == 0 and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                if expr_id not in qvar_ids and expr_id not in keep_var_ids:
                    projection_vars.append(expr)
            else:
                stack.extend(expr.children())
    return projection_vars


@dataclass
class _AffineExpr:
    coeffs: Dict[int, int] = field(default_factory=dict)
    symbols: Dict[int, z3.ArithRef] = field(default_factory=dict)
    const: int = 0

    def add_scaled(self, other: "_AffineExpr", scale: int) -> "_AffineExpr":
        coeffs = dict(self.coeffs)
        symbols = dict(self.symbols)
        for var_id, coeff in other.coeffs.items():
            new_coeff = coeffs.get(var_id, 0) + scale * coeff
            if new_coeff == 0:
                coeffs.pop(var_id, None)
                symbols.pop(var_id, None)
            else:
                coeffs[var_id] = new_coeff
                symbols[var_id] = other.symbols[var_id]
        return _AffineExpr(
            coeffs=coeffs,
            symbols=symbols,
            const=self.const + scale * other.const,
        )

    def scale(self, factor: int) -> "_AffineExpr":
        if factor == 0:
            return _AffineExpr()
        return _AffineExpr(
            coeffs={var_id: coeff * factor for var_id, coeff in self.coeffs.items()},
            symbols=dict(self.symbols),
            const=self.const * factor,
        )

    def without_var(self, var_id: int) -> "_AffineExpr":
        coeffs = dict(self.coeffs)
        symbols = dict(self.symbols)
        coeffs.pop(var_id, None)
        symbols.pop(var_id, None)
        return _AffineExpr(coeffs=coeffs, symbols=symbols, const=self.const)

    def to_expr(self) -> z3.ArithRef:
        parts: List[z3.ArithRef] = []
        for key in sorted(self.coeffs):
            coeff = self.coeffs[key]
            sym = self.symbols[key]
            if coeff == 1:
                parts.append(sym)
            elif coeff == -1:
                parts.append(-sym)
            else:
                parts.append(cast(z3.ArithRef, z3.IntVal(coeff) * sym))
        if self.const != 0 or not parts:
            parts.append(z3.IntVal(self.const))
        if len(parts) == 1:
            return cast(z3.ArithRef, z3.simplify(parts[0]))
        return cast(z3.ArithRef, z3.simplify(z3.Sum(parts)))


@dataclass(frozen=True)
class _AtomInfo:
    coeff: int
    rest: _AffineExpr
    op: str
    modulus: Optional[int] = None


@dataclass(frozen=True)
class _CandidateCongruence:
    coeff_sign: int
    rest: z3.ArithRef
    modulus: int


def _unsupported(message: str) -> None:
    raise ValueError(f"Unsupported Cooper QE fragment: {message}")


def _require_int_var(var: z3.ExprRef) -> None:
    if not z3.is_const(var) or not z3.is_int(var):
        _unsupported("quantified and kept variables must be Int constants")


def _parse_affine_term(expr: z3.ExprRef) -> _AffineExpr:
    if not z3.is_expr(expr):
        _unsupported("non-Z3 term encountered")
    if z3.is_quantifier(expr):
        _unsupported("nested quantifiers are not supported")
    if not z3.is_int(expr):
        _unsupported("mixed or non-Int arithmetic is not supported")
    if z3.is_int_value(expr):
        return _AffineExpr(const=cast(z3.IntNumRef, expr).as_long())
    if z3.is_const(expr):
        if expr.decl().kind() != z3.Z3_OP_UNINTERPRETED:
            _unsupported("casts or non-standard integer constants are not supported")
        expr = cast(z3.ArithRef, expr)
        return _AffineExpr(
            coeffs={expr.get_id(): 1},
            symbols={expr.get_id(): expr},
        )

    kind = expr.decl().kind()
    children = expr.children()

    if kind == z3.Z3_OP_ADD:
        acc = _AffineExpr()
        for child in children:
            acc = acc.add_scaled(_parse_affine_term(child), 1)
        return acc
    if kind == z3.Z3_OP_SUB:
        acc = _parse_affine_term(children[0])
        for child in children[1:]:
            acc = acc.add_scaled(_parse_affine_term(child), -1)
        return acc
    if kind == z3.Z3_OP_UMINUS:
        return _parse_affine_term(children[0]).scale(-1)
    if kind == z3.Z3_OP_MUL:
        if len(children) != 2:
            _unsupported("non-binary multiplication is not supported")
        left, right = children
        if z3.is_int_value(left):
            return _parse_affine_term(right).scale(cast(z3.IntNumRef, left).as_long())
        if z3.is_int_value(right):
            return _parse_affine_term(left).scale(cast(z3.IntNumRef, right).as_long())
        _unsupported("nonlinear arithmetic is not supported")
    if kind in {
        z3.Z3_OP_MOD,
        z3.Z3_OP_IDIV,
        z3.Z3_OP_DIV,
        z3.Z3_OP_TO_REAL,
        z3.Z3_OP_TO_INT,
        z3.Z3_OP_POWER,
    }:
        _unsupported("input mod/div/cast/power terms are not supported")
    if expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        _unsupported("uninterpreted functions are not supported")
    _unsupported(f"unsupported integer term operator: {expr.decl().name()}")
    raise AssertionError("unreachable")


def _comparison_from_term(term: z3.ArithRef, op: str) -> z3.BoolRef:
    zero = z3.IntVal(0)
    if op == "<=":
        return cast(z3.BoolRef, term <= zero)
    if op == "<":
        return cast(z3.BoolRef, term < zero)
    if op == ">=":
        return cast(z3.BoolRef, term >= zero)
    if op == ">":
        return cast(z3.BoolRef, term > zero)
    if op == "==":
        return cast(z3.BoolRef, term == zero)
    _unsupported(f"unsupported comparison operator: {op}")
    raise AssertionError("unreachable")


def _congruence_from_term(term: z3.ArithRef, modulus: int) -> z3.BoolRef:
    if modulus <= 0:
        _unsupported("modulus must be positive")
    return cast(z3.BoolRef, z3.simplify((term % z3.IntVal(modulus)) == 0))


def _expr_plus_const(expr: z3.ArithRef, delta: int) -> z3.ArithRef:
    if delta == 0:
        return cast(z3.ArithRef, z3.simplify(expr))
    return cast(z3.ArithRef, z3.simplify(expr + z3.IntVal(delta)))


def _as_atom_info(literal: z3.BoolRef, var_id: int) -> _AtomInfo:
    if z3.is_not(literal):
        child = literal.arg(0)
        child = cast(z3.BoolRef, child)
        op = ""
        if z3.is_eq(child):
            _unsupported("!= is not supported in the Cooper QE path")
        child_kind = child.decl().kind()
        if child_kind == z3.Z3_OP_LE:
            op = ">"
        elif child_kind == z3.Z3_OP_LT:
            op = ">="
        elif child_kind == z3.Z3_OP_GE:
            op = "<"
        elif child_kind == z3.Z3_OP_GT:
            op = "<="
        else:
            _unsupported("unexpected negated literal after NNF normalization")
        lhs = cast(z3.ArithRef, child.arg(0))
        rhs = cast(z3.ArithRef, child.arg(1))
        affine = _parse_affine_term(cast(z3.ExprRef, lhs - rhs))
        coeff = affine.coeffs.get(var_id, 0)
        return _AtomInfo(coeff=coeff, rest=affine.without_var(var_id), op=op)
    if not z3.is_app(literal) or literal.num_args() != 2:
        _unsupported("only affine arithmetic atoms are supported")
    if not all(z3.is_int(arg) for arg in literal.children()):
        _unsupported("mixed or non-Int comparisons are not supported")

    kind = literal.decl().kind()
    if kind == z3.Z3_OP_EQ:
        left = literal.arg(0)
        right = literal.arg(1)
        mod_side = None
        zero_side = None
        if z3.is_app(left) and left.decl().kind() == z3.Z3_OP_MOD and z3.is_int_value(right):
            mod_side = cast(z3.ArithRef, left)
            zero_side = cast(z3.IntNumRef, right)
        elif z3.is_app(right) and right.decl().kind() == z3.Z3_OP_MOD and z3.is_int_value(left):
            mod_side = cast(z3.ArithRef, right)
            zero_side = cast(z3.IntNumRef, left)
        if mod_side is not None and zero_side is not None and zero_side.as_long() == 0:
            modulus_expr = mod_side.arg(1)
            if not z3.is_int_value(modulus_expr):
                _unsupported("symbolic moduli are not supported")
            modulus = cast(z3.IntNumRef, modulus_expr).as_long()
            if modulus <= 0:
                _unsupported("modulus must be positive")
            affine = _parse_affine_term(cast(z3.ExprRef, mod_side.arg(0)))
            coeff = affine.coeffs.get(var_id, 0)
            return _AtomInfo(
                coeff=coeff,
                rest=affine.without_var(var_id),
                op="congr",
                modulus=modulus,
            )

    op = ""
    if kind == z3.Z3_OP_LE:
        op = "<="
    elif kind == z3.Z3_OP_LT:
        op = "<"
    elif kind == z3.Z3_OP_GE:
        op = ">="
    elif kind == z3.Z3_OP_GT:
        op = ">"
    elif kind == z3.Z3_OP_EQ:
        op = "=="
    elif kind == z3.Z3_OP_DISTINCT:
        _unsupported("!= is not supported in the Cooper QE path")
    else:
        _unsupported(f"unsupported atom kind: {literal.decl().name()}")

    lhs = cast(z3.ArithRef, literal.arg(0))
    rhs = cast(z3.ArithRef, literal.arg(1))
    affine = _parse_affine_term(cast(z3.ExprRef, lhs - rhs))
    coeff = affine.coeffs.get(var_id, 0)
    return _AtomInfo(coeff=coeff, rest=affine.without_var(var_id), op=op)


def _nnf_to_dnf_cubes(expr: z3.BoolRef) -> List[List[z3.BoolRef]]:
    if z3.is_true(expr):
        return [[]]
    if z3.is_false(expr):
        return []
    if z3.is_and(expr):
        cubes: List[List[z3.BoolRef]] = [[]]
        for child in expr.children():
            child_cubes = _nnf_to_dnf_cubes(cast(z3.BoolRef, child))
            next_cubes: List[List[z3.BoolRef]] = []
            for left in cubes:
                for right in child_cubes:
                    next_cubes.append(left + right)
            cubes = next_cubes
        return cubes
    if z3.is_or(expr):
        cubes: List[List[z3.BoolRef]] = []
        for child in expr.children():
            cubes.extend(_nnf_to_dnf_cubes(cast(z3.BoolRef, child)))
        return cubes
    return [[expr]]


def _candidate_congruence_holds(
    candidate: z3.ArithRef, congruence: _CandidateCongruence
) -> z3.BoolRef:
    if congruence.coeff_sign > 0:
        term = cast(z3.ArithRef, z3.simplify(candidate + congruence.rest))
    else:
        term = cast(z3.ArithRef, z3.simplify(congruence.rest - candidate))
    return _congruence_from_term(term, congruence.modulus)


def _eliminate_unit_interval(
    lower_bounds: Sequence[z3.ArithRef],
    upper_bounds: Sequence[z3.ArithRef],
    *,
    search_period: int,
    congruences: Sequence[_CandidateCongruence],
) -> z3.BoolRef:
    if not lower_bounds or not upper_bounds:
        return z3.BoolVal(True)

    branches: List[z3.BoolRef] = []
    deltas = [0] if search_period == 1 else list(range(search_period))
    for index, lower in enumerate(lower_bounds):
        max_guard = [lower >= other for j, other in enumerate(lower_bounds) if j != index]
        residue_branches: List[z3.BoolRef] = []
        for delta in deltas:
            candidate = _expr_plus_const(lower, delta)
            upper_checks = [candidate <= upper for upper in upper_bounds]
            congruence_checks = [
                _candidate_congruence_holds(candidate, congruence)
                for congruence in congruences
            ]
            residue_branches.append(
                cast(
                    z3.BoolRef,
                    z3.And(*(upper_checks + congruence_checks)),
                )
            )
        branches.append(
            cast(z3.BoolRef, z3.And(*(max_guard + [z3.Or(*residue_branches)])))
        )
    return cast(z3.BoolRef, z3.simplify(z3.Or(*branches)))


def _bounds_from_scaled_atom(
    coeff_sign: int,
    scaled_rest: z3.ArithRef,
    op: str,
) -> Tuple[List[z3.ArithRef], List[z3.ArithRef]]:
    lower_bounds: List[z3.ArithRef] = []
    upper_bounds: List[z3.ArithRef] = []

    if coeff_sign > 0:
        if op == "<=":
            upper_bounds.append(_expr_plus_const(-scaled_rest, 0))
        elif op == "<":
            upper_bounds.append(_expr_plus_const(-scaled_rest, -1))
        elif op == ">=":
            lower_bounds.append(_expr_plus_const(-scaled_rest, 0))
        elif op == ">":
            lower_bounds.append(_expr_plus_const(-scaled_rest, 1))
        elif op == "==":
            exact = _expr_plus_const(-scaled_rest, 0)
            lower_bounds.append(exact)
            upper_bounds.append(exact)
    else:
        if op == "<=":
            lower_bounds.append(_expr_plus_const(scaled_rest, 0))
        elif op == "<":
            lower_bounds.append(_expr_plus_const(scaled_rest, 1))
        elif op == ">=":
            upper_bounds.append(_expr_plus_const(scaled_rest, 0))
        elif op == ">":
            upper_bounds.append(_expr_plus_const(scaled_rest, -1))
        elif op == "==":
            exact = _expr_plus_const(scaled_rest, 0)
            lower_bounds.append(exact)
            upper_bounds.append(exact)

    return lower_bounds, upper_bounds


def _eliminate_cube_with_equality(cube: Sequence[z3.BoolRef], var: z3.ExprRef) -> z3.BoolRef:
    var_id = var.get_id()
    pivot_literal: Optional[z3.BoolRef] = None
    pivot_atom: Optional[_AtomInfo] = None
    for literal in cube:
        atom = _as_atom_info(literal, var_id)
        if atom.coeff != 0 and atom.op == "==":
            pivot_literal = literal
            pivot_atom = atom
            break

    if pivot_literal is None or pivot_atom is None:
        return z3.BoolVal(False)

    a = pivot_atom.coeff
    sign = 1 if a > 0 else -1
    abs_a = abs(a)
    pivot_rest = pivot_atom.rest.to_expr()

    result_literals: List[z3.BoolRef] = [
        cast(z3.BoolRef, z3.simplify((pivot_rest % z3.IntVal(abs_a)) == 0))
    ]

    for literal in cube:
        if z3.eq(literal, pivot_literal):
            continue
        atom = _as_atom_info(literal, var_id)
        if atom.op == "congr":
            if atom.modulus is None:
                _unsupported("missing modulus on congruence atom")
            atom_modulus = cast(int, atom.modulus)
            if atom.coeff == 0:
                result_literals.append(_congruence_from_term(atom.rest.to_expr(), atom_modulus))
                continue
            left_term = cast(z3.ArithRef, z3.IntVal(-atom.coeff) * pivot_rest)
            right_term = cast(z3.ArithRef, z3.IntVal(a) * atom.rest.to_expr())
            transformed = cast(
                z3.ArithRef,
                z3.simplify(left_term + right_term),
            )
            result_literals.append(_congruence_from_term(transformed, abs_a * atom_modulus))
            continue
        if atom.coeff == 0:
            result_literals.append(_comparison_from_term(atom.rest.to_expr(), atom.op))
            continue
        left_term = cast(z3.ArithRef, z3.IntVal(-sign * atom.coeff) * pivot_rest)
        right_term = cast(z3.ArithRef, z3.IntVal(abs_a) * atom.rest.to_expr())
        transformed = cast(
            z3.ArithRef,
            z3.simplify(left_term + right_term),
        )
        result_literals.append(
            _comparison_from_term(transformed, atom.op)
        )

    return cast(z3.BoolRef, z3.simplify(z3.And(*result_literals)))


def _eliminate_cube_without_equality(cube: Sequence[z3.BoolRef], var: z3.ExprRef) -> z3.BoolRef:
    var_id = var.get_id()
    atoms = [_as_atom_info(literal, var_id) for literal in cube]

    coeffs = [abs(atom.coeff) for atom in atoms if atom.coeff != 0]
    if not coeffs:
        if not cube:
            return z3.BoolVal(True)
        return cast(z3.BoolRef, z3.simplify(z3.And(*cube)))

    modulus = 1
    for coeff in coeffs:
        modulus = _lcm(modulus, coeff)

    lower_bounds: List[z3.ArithRef] = []
    upper_bounds: List[z3.ArithRef] = []
    independent_literals: List[z3.BoolRef] = []
    candidate_congruences: List[_CandidateCongruence] = []

    if modulus > 1:
        candidate_congruences.append(
            _CandidateCongruence(
                coeff_sign=1,
                rest=z3.IntVal(0),
                modulus=modulus,
            )
        )

    search_period = modulus

    for atom in atoms:
        if atom.op == "congr":
            if atom.modulus is None:
                _unsupported("missing modulus on congruence atom")
            atom_modulus = cast(int, atom.modulus)
            if atom.coeff == 0:
                independent_literals.append(
                    _congruence_from_term(atom.rest.to_expr(), atom_modulus)
                )
                continue

            scale = modulus // abs(atom.coeff)
            scaled_rest = cast(
                z3.ArithRef, z3.simplify(z3.IntVal(scale) * atom.rest.to_expr())
            )
            scaled_modulus = scale * atom_modulus
            search_period = _lcm(search_period, scaled_modulus)
            candidate_congruences.append(
                _CandidateCongruence(
                    coeff_sign=1 if atom.coeff > 0 else -1,
                    rest=scaled_rest,
                    modulus=scaled_modulus,
                )
            )
            continue
        if atom.coeff == 0:
            independent_literals.append(_comparison_from_term(atom.rest.to_expr(), atom.op))
            continue
        scale = modulus // abs(atom.coeff)
        scaled_rest = cast(
            z3.ArithRef, z3.simplify(z3.IntVal(scale) * atom.rest.to_expr())
        )
        atom_lower, atom_upper = _bounds_from_scaled_atom(
            1 if atom.coeff > 0 else -1,
            scaled_rest,
            atom.op,
        )
        lower_bounds.extend(atom_lower)
        upper_bounds.extend(atom_upper)

    interval = _eliminate_unit_interval(
        lower_bounds,
        upper_bounds,
        search_period=search_period,
        congruences=candidate_congruences,
    )
    return cast(z3.BoolRef, z3.simplify(z3.And(*(independent_literals + [interval]))))


def _eliminate_cube(cube: Sequence[z3.BoolRef], var: z3.ExprRef) -> z3.BoolRef:
    var_id = var.get_id()
    for literal in cube:
        atom = _as_atom_info(literal, var_id)
        if atom.coeff != 0 and atom.op == "==":
            return _eliminate_cube_with_equality(cube, var)
    return _eliminate_cube_without_equality(cube, var)


def _ensure_supported_formula(phi: z3.BoolRef) -> None:
    if not z3.is_bool(phi):
        _unsupported("phi must be a Boolean formula")

    stack = [phi]
    while stack:
        expr = stack.pop()
        if z3.is_quantifier(expr):
            _unsupported("nested quantifiers are not supported")
        if z3.is_true(expr) or z3.is_false(expr):
            continue
        if (
            z3.is_and(expr)
            or z3.is_or(expr)
            or z3.is_not(expr)
            or z3.is_implies(expr)
            or expr.decl().kind() == z3.Z3_OP_XOR
        ):
            stack.extend(cast(Iterable[z3.BoolRef], expr.children()))
            continue
        if expr.decl().kind() == z3.Z3_OP_IFF:
            stack.extend(cast(Iterable[z3.BoolRef], expr.children()))
            continue
        if not z3.is_app(expr):
            _unsupported("unexpected non-application Boolean term")
        if expr.num_args() == 0:
            _unsupported("Boolean variables are not supported in the Cooper QE path")
        if expr.decl().kind() == z3.Z3_OP_DISTINCT:
            _unsupported("!= is not supported in the Cooper QE path")
        if expr.num_args() != 2:
            _unsupported("only binary affine arithmetic atoms are supported")
        _parse_affine_term(expr.arg(0))
        _parse_affine_term(expr.arg(1))


def _eliminate_one_var(phi: z3.BoolRef, var: z3.ExprRef) -> z3.BoolRef:
    nnf = cast(z3.BoolRef, z3.Tactic("nnf")(phi).as_expr())
    cubes = _nnf_to_dnf_cubes(nnf)
    projected_cubes = [_eliminate_cube(cube, var) for cube in cubes]
    if not projected_cubes:
        return z3.BoolVal(False)
    return cast(z3.BoolRef, z3.simplify(z3.Or(*projected_cubes)))


def qelim_exists_lia_cooper(
    phi: Any, qvars: Any, *, keep_vars: Optional[Any] = None
) -> z3.BoolRef:
    """Eliminate existentially quantified Int variables from a supported LIA formula.

    The supported fragment is quantifier-free Presburger-style LIA over Ints only,
    with affine arithmetic atoms and Boolean structure. Unsupported constructs fail
    fast with ``ValueError``.

    When ``keep_vars`` is provided, every free variable in ``phi`` that is not
    explicitly kept is projected away together with ``qvars``, matching
    ``qelim_exists_lme`` semantics.
    """

    bool_phi = cast(z3.BoolRef, phi)
    normalized_qvars = _normalize_vars(qvars)
    normalized_keep_vars = _normalize_vars(keep_vars) if keep_vars is not None else None

    for var in normalized_qvars:
        _require_int_var(var)
    if normalized_keep_vars is not None:
        for var in normalized_keep_vars:
            _require_int_var(var)

    projection_vars = _get_projection_vars(
        bool_phi,
        normalized_qvars,
        normalized_keep_vars,
    )
    qvar_ids = {var.get_id() for var in normalized_qvars}
    ordered_projection_vars = [
        var for var in projection_vars if var.get_id() not in qvar_ids
    ] + list(normalized_qvars)
    for var in ordered_projection_vars:
        _require_int_var(var)

    _ensure_supported_formula(bool_phi)

    result = cast(z3.BoolRef, z3.simplify(bool_phi))
    for var in ordered_projection_vars:
        result = _eliminate_one_var(result, var)
    return cast(z3.BoolRef, z3.simplify(result))
