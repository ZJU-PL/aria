#!/usr/bin/env python3
"""Polynomial IR and normalization utilities for finite-field formulas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .ff_ast import (
    FieldAdd,
    FieldConst,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldSub,
    FieldVar,
    infer_field_modulus,
)

Monomial = Tuple[Tuple[str, int], ...]


def _normalize_monomial(monomial: Monomial) -> Monomial:
    merged: Dict[str, int] = {}
    for var_name, exponent in monomial:
        if exponent <= 0:
            continue
        merged[var_name] = merged.get(var_name, 0) + exponent
    return tuple(sorted(merged.items()))


def _multiply_monomials(left: Monomial, right: Monomial) -> Monomial:
    return _normalize_monomial(left + right)


@dataclass(frozen=True)
class PolynomialPartition:
    """A variable-connected component of polynomial assertions."""

    modulus: int
    assertion_indices: Tuple[int, ...]
    variables: Tuple[str, ...]


class FFPolynomial:
    """Sparse polynomial with coefficients in GF(p)."""

    def __init__(self, modulus: int, terms: Optional[Dict[Monomial, int]] = None):
        if modulus <= 1:
            raise ValueError("polynomial modulus must be > 1")
        self.modulus = modulus
        normalized: Dict[Monomial, int] = {}
        for monomial, coeff in (terms or {}).items():
            reduced = coeff % modulus
            if reduced == 0:
                continue
            key = _normalize_monomial(monomial)
            normalized[key] = (normalized.get(key, 0) + reduced) % modulus
            if normalized[key] == 0:
                del normalized[key]
        self.terms = normalized

    @classmethod
    def zero(cls, modulus: int) -> "FFPolynomial":
        return cls(modulus, {})

    @classmethod
    def one(cls, modulus: int) -> "FFPolynomial":
        return cls(modulus, {(): 1})

    @classmethod
    def const(cls, value: int, modulus: int) -> "FFPolynomial":
        return cls(modulus, {(): value})

    @classmethod
    def var(cls, name: str, modulus: int) -> "FFPolynomial":
        return cls(modulus, {((name, 1),): 1})

    def copy(self) -> "FFPolynomial":
        return FFPolynomial(self.modulus, dict(self.terms))

    def is_zero(self) -> bool:
        return not self.terms

    def is_constant(self) -> bool:
        return all(not monomial for monomial in self.terms)

    def constant_term(self) -> int:
        return self.terms.get((), 0)

    def variables(self) -> Set[str]:
        names: Set[str] = set()
        for monomial in self.terms:
            for var_name, _exponent in monomial:
                names.add(var_name)
        return names

    def degree(self) -> int:
        best = 0
        for monomial in self.terms:
            best = max(best, sum(exponent for _name, exponent in monomial))
        return best

    def term_count(self) -> int:
        return len(self.terms)

    def add(self, other: "FFPolynomial") -> "FFPolynomial":
        self._check_compatibility(other)
        merged = dict(self.terms)
        for monomial, coeff in other.terms.items():
            merged[monomial] = (merged.get(monomial, 0) + coeff) % self.modulus
            if merged[monomial] == 0:
                del merged[monomial]
        return FFPolynomial(self.modulus, merged)

    def neg(self) -> "FFPolynomial":
        return FFPolynomial(
            self.modulus,
            {monomial: (-coeff) % self.modulus for monomial, coeff in self.terms.items()},
        )

    def sub(self, other: "FFPolynomial") -> "FFPolynomial":
        return self.add(other.neg())

    def mul(self, other: "FFPolynomial") -> "FFPolynomial":
        self._check_compatibility(other)
        if self.is_zero() or other.is_zero():
            return FFPolynomial.zero(self.modulus)
        product: Dict[Monomial, int] = {}
        for left_monomial, left_coeff in self.terms.items():
            for right_monomial, right_coeff in other.terms.items():
                monomial = _multiply_monomials(left_monomial, right_monomial)
                coeff = (left_coeff * right_coeff) % self.modulus
                product[monomial] = (product.get(monomial, 0) + coeff) % self.modulus
                if product[monomial] == 0:
                    del product[monomial]
        return FFPolynomial(self.modulus, product)

    def pow(self, exponent: int) -> "FFPolynomial":
        if exponent < 0:
            raise ValueError("polynomial exponent must be non-negative")
        result = FFPolynomial.one(self.modulus)
        base = self
        exp = exponent
        while exp > 0:
            if exp & 1:
                result = result.mul(base)
            exp >>= 1
            if exp:
                base = base.mul(base)
        return result

    def to_expr(self) -> FieldExpr:
        if self.is_zero():
            return FieldConst(0, self.modulus)

        terms: List[FieldExpr] = []
        for monomial in sorted(self.terms):
            coeff = self.terms[monomial]
            if not monomial:
                terms.append(FieldConst(coeff, self.modulus))
                continue

            factors: List[FieldExpr] = []
            if coeff != 1:
                factors.append(FieldConst(coeff, self.modulus))
            for var_name, exponent in monomial:
                var_expr: FieldExpr = FieldVar(var_name)
                if exponent != 1:
                    var_expr = FieldPow(var_expr, exponent)
                factors.append(var_expr)

            if not factors:
                terms.append(FieldConst(1, self.modulus))
            elif len(factors) == 1:
                terms.append(factors[0])
            else:
                terms.append(FieldMul(*factors))

        if len(terms) == 1:
            return terms[0]
        return FieldAdd(*terms)

    def _check_compatibility(self, other: "FFPolynomial") -> None:
        if self.modulus != other.modulus:
            raise ValueError(
                "polynomial modulus mismatch: %d vs %d"
                % (self.modulus, other.modulus)
            )


def polynomial_from_expr(
    expr: FieldExpr, variables: Dict[str, str]
) -> Optional[FFPolynomial]:
    """Lower an arithmetic field expression to sparse polynomial IR."""
    modulus = infer_field_modulus(expr, variables)
    if modulus is None:
        return None

    if isinstance(expr, FieldConst):
        return FFPolynomial.const(expr.value, modulus)
    if isinstance(expr, FieldVar):
        return FFPolynomial.var(expr.name, modulus)
    if isinstance(expr, FieldAdd):
        result = FFPolynomial.zero(modulus)
        for arg in expr.args:
            arg_poly = polynomial_from_expr(arg, variables)
            if arg_poly is None:
                return None
            result = result.add(arg_poly)
        return result
    if isinstance(expr, FieldSub):
        result = polynomial_from_expr(expr.args[0], variables)
        if result is None:
            return None
        for arg in expr.args[1:]:
            arg_poly = polynomial_from_expr(arg, variables)
            if arg_poly is None:
                return None
            result = result.sub(arg_poly)
        return result
    if isinstance(expr, FieldNeg):
        arg_poly = polynomial_from_expr(expr.arg, variables)
        return None if arg_poly is None else arg_poly.neg()
    if isinstance(expr, FieldMul):
        result = FFPolynomial.one(modulus)
        for arg in expr.args:
            arg_poly = polynomial_from_expr(arg, variables)
            if arg_poly is None:
                return None
            result = result.mul(arg_poly)
        return result
    if isinstance(expr, FieldPow):
        base_poly = polynomial_from_expr(expr.base, variables)
        return None if base_poly is None else base_poly.pow(expr.exponent)
    return None


def polynomial_from_equality(
    assertion: FieldExpr, variables: Dict[str, str]
) -> Optional[FFPolynomial]:
    """Return polynomial ``lhs - rhs`` for a field equality."""
    if not isinstance(assertion, FieldEq):
        return None
    left_poly = polynomial_from_expr(assertion.left, variables)
    right_poly = polynomial_from_expr(assertion.right, variables)
    if left_poly is None or right_poly is None:
        return None
    return left_poly.sub(right_poly)


def partition_polynomial_assertions(
    assertions: Sequence[FieldExpr], variables: Dict[str, str]
) -> List[PolynomialPartition]:
    """Partition polynomial equalities by shared variables within each field."""
    by_modulus: Dict[int, List[Tuple[int, Set[str]]]] = {}
    for idx, assertion in enumerate(assertions):
        poly = polynomial_from_equality(assertion, variables)
        if poly is None:
            continue
        by_modulus.setdefault(poly.modulus, []).append((idx, poly.variables()))

    partitions: List[PolynomialPartition] = []
    for modulus, entries in by_modulus.items():
        if not entries:
            continue
        parent = list(range(len(entries)))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        seen_vars: Dict[str, int] = {}
        for local_idx, (_assertion_idx, var_names) in enumerate(entries):
            for var_name in var_names:
                prev = seen_vars.get(var_name)
                if prev is None:
                    seen_vars[var_name] = local_idx
                    continue
                union(local_idx, prev)

        groups: Dict[int, Dict[str, Set[object]]] = {}
        for local_idx, (assertion_idx, var_names) in enumerate(entries):
            root = find(local_idx)
            group = groups.setdefault(root, {"indices": set(), "variables": set()})
            group["indices"].add(assertion_idx)
            group["variables"].update(var_names)

        for group in groups.values():
            partitions.append(
                PolynomialPartition(
                    modulus=modulus,
                    assertion_indices=tuple(sorted(group["indices"])),
                    variables=tuple(sorted(group["variables"])),
                )
            )

    partitions.sort(
        key=lambda part: (part.modulus, len(part.assertion_indices), part.assertion_indices)
    )
    return partitions
