#!/usr/bin/env python3
"""Lightweight algebraic reasoning for finite-field polynomial equalities.

The module provides exact local reasoning only. Every learned consequence is
meant to be sound in GF(p) and cheap enough to use inside the solver's CEGAR
loop without introducing a heavyweight external CAS dependency.

The current reasoning tiers are:
    - one-assertion rewrites and lemmas for simple algebraic patterns;
    - affine elimination on small connected polynomial partitions;
    - bounded nonlinear model enumeration on very small partitions/moduli;
    - memoization of partition results so repeated refinement does not redo the
      same local work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ff_ast import (
    BoolConst,
    BoolOr,
    FieldAdd,
    FieldConst,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldPow,
    FieldVar,
    infer_field_modulus,
)
from .ff_poly import (
    FFPolynomial,
    PolynomialPartition,
    partition_polynomial_assertions,
    polynomial_from_equality,
)
from .ff_ir import expr_key


@dataclass(frozen=True)
class AlgebraicLemma:
    """An exact finite-field lemma derived from local algebraic reasoning."""

    kind: str
    expr: FieldExpr


class FFLocalAlgebraicReasoner:
    """Derive exact consequences from small polynomial equalities.

    This class is intentionally conservative: it only returns lemmas that are
    justified by exact algebra over GF(p). The bounded nonlinear path is a
    complete local search over the selected partition when enabled by the size
    limits; otherwise the reasoner declines to solve that partition.
    """

    def __init__(
        self,
        max_zero_product_arity: int = 4,
        max_affine_partition_eqs: int = 6,
        max_affine_partition_vars: int = 6,
        max_nonlinear_partition_eqs: int = 4,
        max_nonlinear_partition_vars: int = 4,
        max_nonlinear_modulus: int = 257,
        max_nonlinear_search_space: int = 4096,
        max_nonlinear_work_budget: int = 8192,
        rootset_budget: int = 4,
    ):
        self.max_zero_product_arity = max(2, max_zero_product_arity)
        self.max_affine_partition_eqs = max(2, max_affine_partition_eqs)
        self.max_affine_partition_vars = max(1, max_affine_partition_vars)
        self.max_nonlinear_partition_eqs = max(1, max_nonlinear_partition_eqs)
        self.max_nonlinear_partition_vars = max(1, max_nonlinear_partition_vars)
        self.max_nonlinear_modulus = max(2, max_nonlinear_modulus)
        self.max_nonlinear_search_space = max(2, max_nonlinear_search_space)
        self.max_nonlinear_work_budget = max(16, max_nonlinear_work_budget)
        self.rootset_budget = max(2, rootset_budget)
        self._partition_cache: Dict[Tuple[object, ...], List[AlgebraicLemma]] = {}
        self._stats: Dict[str, int] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset cumulative stats for one outer solving run."""
        self._stats = {
            "partition_cache_hits": 0,
            "partition_cache_misses": 0,
            "partition_solver_calls": 0,
            "partition_solver_enumerations": 0,
            "partition_solver_search_nodes": 0,
            "partition_solver_budget_abort": 0,
            "partition_solver_small_space_skip": 0,
        }

    def stats(self) -> Dict[str, int]:
        """Return reasoner stats accumulated so far."""
        return dict(self._stats)

    def derive_lemmas(
        self, assertion: FieldExpr, variables: Dict[str, str]
    ) -> List[AlgebraicLemma]:
        """Return exact lemmas implied by one polynomial assertion."""
        poly = polynomial_from_equality(assertion, variables)
        if poly is None:
            return []

        lemmas: List[AlgebraicLemma] = []
        contradiction = self._constant_contradiction(poly)
        if contradiction is not None and contradiction.kind != "tautology":
            lemmas.append(contradiction)

        zero_product = self._zero_product_lemma(assertion, variables)
        if zero_product is not None:
            lemmas.append(zero_product)

        power_zero = self._power_zero_lemma(assertion, variables)
        if power_zero is not None:
            lemmas.append(power_zero)

        linear = self._univariate_linear_lemma(poly)
        if linear is not None:
            lemmas.append(linear)

        monomial = self._univariate_monomial_zero_lemma(poly)
        if monomial is not None:
            lemmas.append(monomial)
        return lemmas

    def derive_partition_lemmas(
        self, assertions: Sequence[FieldExpr], variables: Dict[str, str]
    ) -> List[AlgebraicLemma]:
        """Return exact lemmas from small polynomial partitions.

        The method first partitions the polynomial assertions by shared
        variables, then tries the affine solver, then the bounded nonlinear
        solver, caching the resulting lemma list per partition signature.
        """
        lemmas: List[AlgebraicLemma] = []
        seen = set()
        for partition in partition_polynomial_assertions(assertions, variables):
            cache_key = self._partition_cache_key(partition, assertions, variables)
            cached = self._partition_cache.get(cache_key)
            if cached is not None:
                self._stats["partition_cache_hits"] += 1
                candidates = cached
            else:
                self._stats["partition_cache_misses"] += 1
                candidates = []
                candidates.extend(
                    self._derive_affine_partition_lemmas(
                        partition, assertions, variables
                    )
                )
                candidates.extend(
                    self._derive_exact_partition_lemmas(
                        partition, assertions, variables
                    )
                )
                self._partition_cache[cache_key] = list(candidates)
            for lemma in candidates:
                key = _lemma_key(lemma)
                if key in seen:
                    continue
                seen.add(key)
                lemmas.append(lemma)
        return lemmas

    def rewrite_structured_assertion(
        self, assertion: FieldExpr, variables: Dict[str, str]
    ) -> FieldExpr:
        """Rewrite exact sparse algebraic patterns into more explicit form."""
        for lemma in self.derive_lemmas(assertion, variables):
            if lemma.kind in (
                "constant-contradiction",
                "power-zero",
            ):
                return lemma.expr
        return assertion

    def _constant_contradiction(
        self, poly: FFPolynomial
    ) -> Optional[AlgebraicLemma]:
        if poly.is_zero():
            return AlgebraicLemma("tautology", BoolConst(True))
        if poly.is_constant() and poly.constant_term() != 0:
            return AlgebraicLemma("constant-contradiction", BoolConst(False))
        return None

    def _zero_product_lemma(
        self, assertion: FieldExpr, variables: Dict[str, str]
    ) -> Optional[AlgebraicLemma]:
        zero_eq = _match_eq_zero(assertion, variables)
        if zero_eq is None:
            return None
        expr, modulus = zero_eq
        if not isinstance(expr, FieldMul):
            return None

        factors = []
        for factor in expr.args:
            if isinstance(factor, FieldConst):
                if factor.value == 0:
                    return AlgebraicLemma("zero-product", BoolConst(True))
                continue
            factors.append(factor)

        if len(factors) < 2 or len(factors) > self.max_zero_product_arity:
            return None

        zero = FieldConst(0, modulus)
        return AlgebraicLemma(
            "zero-product",
            BoolOr(*[FieldEq(factor, zero) for factor in factors]),
        )

    def _power_zero_lemma(
        self, assertion: FieldExpr, variables: Dict[str, str]
    ) -> Optional[AlgebraicLemma]:
        zero_eq = _match_eq_zero(assertion, variables)
        if zero_eq is None:
            return None
        expr, modulus = zero_eq
        if not isinstance(expr, FieldPow) or expr.exponent <= 0:
            return None
        return AlgebraicLemma(
            "power-zero",
            FieldEq(expr.base, FieldConst(0, modulus)),
        )

    def _univariate_linear_lemma(
        self, poly: FFPolynomial
    ) -> Optional[AlgebraicLemma]:
        if poly.is_zero() or len(poly.variables()) != 1:
            return None
        var_name = next(iter(poly.variables()))
        linear_coeff = poly.terms.get(((var_name, 1),), 0)
        constant_coeff = poly.constant_term()
        if linear_coeff == 0:
            return None
        if len(poly.terms) > (2 if constant_coeff != 0 else 1):
            return None

        inv = pow(linear_coeff, -1, poly.modulus)
        root = (-constant_coeff * inv) % poly.modulus
        return AlgebraicLemma(
            "linear-root",
            FieldEq(FieldVar(var_name), FieldConst(root, poly.modulus)),
        )

    def _univariate_monomial_zero_lemma(
        self, poly: FFPolynomial
    ) -> Optional[AlgebraicLemma]:
        if len(poly.variables()) != 1 or poly.constant_term() != 0 or len(poly.terms) != 1:
            return None
        monomial = next(iter(poly.terms))
        if not monomial:
            return None
        if len(monomial) != 1:
            return None
        var_name, exponent = monomial[0]
        if exponent <= 0:
            return None
        return AlgebraicLemma(
            "monomial-zero",
            FieldEq(
                FieldVar(var_name),
                FieldConst(0, poly.modulus),
            ),
        )

    def _derive_affine_partition_lemmas(
        self,
        partition: PolynomialPartition,
        assertions: Sequence[FieldExpr],
        variables: Dict[str, str],
    ) -> List[AlgebraicLemma]:
        if len(partition.assertion_indices) < 2:
            return []
        if len(partition.assertion_indices) > self.max_affine_partition_eqs:
            return []
        if len(partition.variables) > self.max_affine_partition_vars:
            return []

        var_names = list(partition.variables)
        rows: List[List[int]] = []
        rhs: List[int] = []
        for assertion_idx in partition.assertion_indices:
            poly = polynomial_from_equality(assertions[assertion_idx], variables)
            if poly is None:
                return []
            affine_row = _affine_row(poly, var_names)
            if affine_row is None:
                return []
            coeffs, value = affine_row
            rows.append(coeffs)
            rhs.append(value)

        rref_rows, rref_rhs = _rref_mod(rows, rhs, partition.modulus)
        lemmas: List[AlgebraicLemma] = []
        for coeffs, value in zip(rref_rows, rref_rhs):
            nonzero_cols = [idx for idx, coeff in enumerate(coeffs) if coeff % partition.modulus]
            if not nonzero_cols:
                if value % partition.modulus:
                    return [
                        AlgebraicLemma("affine-contradiction", BoolConst(False))
                    ]
                continue
            if len(nonzero_cols) == 1:
                col = nonzero_cols[0]
                coeff = coeffs[col] % partition.modulus
                root = (value * pow(coeff, -1, partition.modulus)) % partition.modulus
                lemmas.append(
                    AlgebraicLemma(
                        "affine-root",
                        FieldEq(
                            FieldVar(var_names[col]),
                            FieldConst(root, partition.modulus),
                        ),
                    )
                )
                continue
            normalized = _normalize_affine_relation(
                coeffs, value, var_names, partition.modulus
            )
            if normalized is not None:
                lemmas.append(AlgebraicLemma("affine-relation", normalized))
        return lemmas

    def _derive_exact_partition_lemmas(
        self,
        partition: PolynomialPartition,
        assertions: Sequence[FieldExpr],
        variables: Dict[str, str],
    ) -> List[AlgebraicLemma]:
        """Enumerate exact models of a small nonlinear partition when safe.

        The search is only attempted when the partition passes all configured
        guards: number of equations, number of variables, modulus bound, search
        space bound, and explicit work budget.
        """
        self._stats["partition_solver_calls"] += 1
        if len(partition.assertion_indices) > self.max_nonlinear_partition_eqs:
            self._stats["partition_solver_small_space_skip"] += 1
            return []
        if len(partition.variables) == 0:
            return []
        if len(partition.variables) > self.max_nonlinear_partition_vars:
            self._stats["partition_solver_small_space_skip"] += 1
            return []
        if partition.modulus > self.max_nonlinear_modulus:
            self._stats["partition_solver_small_space_skip"] += 1
            return []
        if partition.modulus ** len(partition.variables) > self.max_nonlinear_search_space:
            self._stats["partition_solver_small_space_skip"] += 1
            return []

        polynomials: List[FFPolynomial] = []
        for assertion_idx in partition.assertion_indices:
            poly = polynomial_from_equality(assertions[assertion_idx], variables)
            if poly is None:
                return []
            polynomials.append(poly)

        var_order = _search_variable_order(polynomials, partition.variables)
        models: List[Dict[str, int]] = []
        budget = {"remaining": self.max_nonlinear_work_budget, "aborted": False}
        self._stats["partition_solver_enumerations"] += 1
        self._enumerate_partition_models(
            polynomials,
            partition.modulus,
            var_order,
            0,
            {},
            models,
            budget,
        )
        if budget["aborted"]:
            self._stats["partition_solver_budget_abort"] += 1
            return []

        if not models:
            return [AlgebraicLemma("partition-contradiction", BoolConst(False))]

        lemmas: List[AlgebraicLemma] = []
        for var_name in partition.variables:
            values = sorted({model[var_name] for model in models})
            if len(values) == 1:
                lemmas.append(
                    AlgebraicLemma(
                        "partition-root",
                        FieldEq(FieldVar(var_name), FieldConst(values[0], partition.modulus)),
                    )
                )
            elif len(values) <= self.rootset_budget:
                lemmas.append(
                    AlgebraicLemma(
                        "partition-rootset",
                        BoolOr(
                            *[
                                FieldEq(
                                    FieldVar(var_name),
                                    FieldConst(value, partition.modulus),
                                )
                                for value in values
                            ]
                        ),
                    )
                )

        for left_var in partition.variables:
            for right_var in partition.variables:
                if left_var == right_var:
                    continue
                relation = _infer_affine_model_relation(
                    models, left_var, right_var, partition.modulus
                )
                if relation is None:
                    continue
                coeff, bias = relation
                rhs = _affine_rhs_expr(right_var, coeff, bias, partition.modulus)
                lemmas.append(
                    AlgebraicLemma(
                        "partition-relation",
                        FieldEq(FieldVar(left_var), rhs),
                    )
                )
        return lemmas

    def _enumerate_partition_models(
        self,
        polynomials: Sequence[FFPolynomial],
        modulus: int,
        var_order: Sequence[str],
        depth: int,
        assignment: Dict[str, int],
        models: List[Dict[str, int]],
        budget: Dict[str, int],
    ) -> None:
        """Depth-first search over the local finite-field assignment space."""
        if budget["remaining"] <= 0:
            budget["aborted"] = True
            return
        budget["remaining"] -= 1
        self._stats["partition_solver_search_nodes"] += 1
        if depth >= len(var_order):
            if all(_eval_poly(poly, assignment, modulus) == 0 for poly in polynomials):
                models.append(dict(assignment))
            return

        var_name = var_order[depth]
        for value in range(modulus):
            if budget["remaining"] <= 0:
                budget["aborted"] = True
                break
            assignment[var_name] = value
            if _is_partial_assignment_feasible(polynomials, assignment, modulus):
                self._enumerate_partition_models(
                    polynomials,
                    modulus,
                    var_order,
                    depth + 1,
                    assignment,
                    models,
                    budget,
                )
                if budget["aborted"]:
                    break
        assignment.pop(var_name, None)

    def _partition_cache_key(
        self,
        partition: PolynomialPartition,
        assertions: Sequence[FieldExpr],
        variables: Dict[str, str],
    ) -> Tuple[object, ...]:
        signature = []
        for assertion_idx in partition.assertion_indices:
            poly = polynomial_from_equality(assertions[assertion_idx], variables)
            if poly is None:
                signature.append(("nonpoly", assertion_idx))
                continue
            signature.append(
                (
                    assertion_idx,
                    tuple(sorted((monomial, coeff) for monomial, coeff in poly.terms.items())),
                )
            )
        return (
            partition.modulus,
            partition.variables,
            tuple(signature),
        )


def _match_eq_zero(
    assertion: FieldExpr, variables: Dict[str, str]
) -> Optional[Tuple[FieldExpr, int]]:
    if not isinstance(assertion, FieldEq):
        return None
    left_modulus = infer_field_modulus(assertion.left, variables)
    right_modulus = infer_field_modulus(assertion.right, variables)
    if left_modulus is not None and isinstance(assertion.right, FieldConst):
        if assertion.right.modulus == left_modulus and assertion.right.value == 0:
            return (assertion.left, left_modulus)
    if right_modulus is not None and isinstance(assertion.left, FieldConst):
        if assertion.left.modulus == right_modulus and assertion.left.value == 0:
            return (assertion.right, right_modulus)
    return None


def _affine_row(
    poly: FFPolynomial, var_names: Sequence[str]
) -> Optional[Tuple[List[int], int]]:
    coeffs = [0 for _ in var_names]
    for monomial, coeff in poly.terms.items():
        if not monomial:
            continue
        if len(monomial) != 1:
            return None
        var_name, exponent = monomial[0]
        if exponent != 1:
            return None
        try:
            col = var_names.index(var_name)
        except ValueError:
            return None
        coeffs[col] = (coeffs[col] + coeff) % poly.modulus
    value = (-poly.constant_term()) % poly.modulus
    return (coeffs, value)


def _rref_mod(
    rows: Sequence[Sequence[int]], rhs: Sequence[int], modulus: int
) -> Tuple[List[List[int]], List[int]]:
    matrix = [list(row) for row in rows]
    values = [value % modulus for value in rhs]
    pivot_row = 0
    pivot_col = 0
    while pivot_row < len(matrix) and pivot_col < (len(matrix[0]) if matrix else 0):
        selected = None
        for row_idx in range(pivot_row, len(matrix)):
            if matrix[row_idx][pivot_col] % modulus:
                selected = row_idx
                break
        if selected is None:
            pivot_col += 1
            continue
        if selected != pivot_row:
            matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
            values[pivot_row], values[selected] = values[selected], values[pivot_row]

        pivot = matrix[pivot_row][pivot_col] % modulus
        inv = pow(pivot, -1, modulus)
        matrix[pivot_row] = [(entry * inv) % modulus for entry in matrix[pivot_row]]
        values[pivot_row] = (values[pivot_row] * inv) % modulus

        for row_idx in range(len(matrix)):
            if row_idx == pivot_row:
                continue
            factor = matrix[row_idx][pivot_col] % modulus
            if factor == 0:
                continue
            matrix[row_idx] = [
                (entry - factor * pivot_entry) % modulus
                for entry, pivot_entry in zip(matrix[row_idx], matrix[pivot_row])
            ]
            values[row_idx] = (values[row_idx] - factor * values[pivot_row]) % modulus

        pivot_row += 1
        pivot_col += 1
    return matrix, values


def _normalize_affine_relation(
    coeffs: Sequence[int],
    value: int,
    var_names: Sequence[str],
    modulus: int,
) -> Optional[FieldExpr]:
    leading_col = None
    for idx, coeff in enumerate(coeffs):
        if coeff % modulus:
            leading_col = idx
            break
    if leading_col is None:
        return None
    if coeffs[leading_col] % modulus != 1:
        return None

    rhs_terms: List[FieldExpr] = []
    for idx in range(leading_col + 1, len(coeffs)):
        coeff = coeffs[idx] % modulus
        if coeff == 0:
            continue
        neg_coeff = (-coeff) % modulus
        if neg_coeff == 1:
            rhs_terms.append(FieldVar(var_names[idx]))
        else:
            rhs_terms.append(
                FieldMul(FieldConst(neg_coeff, modulus), FieldVar(var_names[idx]))
            )
    if value % modulus:
        rhs_terms.append(FieldConst(value % modulus, modulus))

    if not rhs_terms:
        rhs_expr: FieldExpr = FieldConst(0, modulus)
    elif len(rhs_terms) == 1:
        rhs_expr = rhs_terms[0]
    else:
        rhs_expr = FieldAdd(*rhs_terms)
    return FieldEq(FieldVar(var_names[leading_col]), rhs_expr)


def _lemma_key(lemma: AlgebraicLemma) -> Tuple[str, str]:
    return (lemma.kind, str(expr_key(lemma.expr)))


def _search_variable_order(
    polynomials: Sequence[FFPolynomial], variables: Sequence[str]
) -> List[str]:
    counts = {var_name: 0 for var_name in variables}
    degrees = {var_name: 0 for var_name in variables}
    for poly in polynomials:
        for monomial, _coeff in poly.terms.items():
            for var_name, exponent in monomial:
                if var_name in counts:
                    counts[var_name] += 1
                    degrees[var_name] = max(degrees[var_name], exponent)
    return sorted(
        variables,
        key=lambda name: (counts.get(name, 0), degrees.get(name, 0), name),
        reverse=True,
    )


def _is_partial_assignment_feasible(
    polynomials: Sequence[FFPolynomial],
    assignment: Dict[str, int],
    modulus: int,
) -> bool:
    for poly in polynomials:
        status = _partial_poly_status(poly, assignment, modulus)
        if status == "nonzero":
            return False
    return True


def _partial_poly_status(
    poly: FFPolynomial, assignment: Dict[str, int], modulus: int
) -> str:
    constant = 0
    has_symbolic_term = False
    for monomial, coeff in poly.terms.items():
        term_coeff = coeff % modulus
        unresolved = False
        for var_name, exponent in monomial:
            value = assignment.get(var_name)
            if value is None:
                unresolved = True
                break
            term_coeff = (term_coeff * pow(value, exponent, modulus)) % modulus
        if unresolved:
            has_symbolic_term = True
            continue
        constant = (constant + term_coeff) % modulus
    if has_symbolic_term:
        return "unknown"
    if constant == 0:
        return "zero"
    return "nonzero"


def _eval_poly(poly: FFPolynomial, assignment: Dict[str, int], modulus: int) -> int:
    total = 0
    for monomial, coeff in poly.terms.items():
        term_coeff = coeff % modulus
        for var_name, exponent in monomial:
            term_coeff = (term_coeff * pow(assignment[var_name], exponent, modulus)) % modulus
        total = (total + term_coeff) % modulus
    return total


def _infer_affine_model_relation(
    models: Sequence[Dict[str, int]],
    left_var: str,
    right_var: str,
    modulus: int,
) -> Optional[Tuple[int, int]]:
    if len(models) < 2:
        return None
    candidate = None
    for idx in range(len(models)):
        for jdx in range(idx + 1, len(models)):
            left_delta = (models[idx][left_var] - models[jdx][left_var]) % modulus
            right_delta = (models[idx][right_var] - models[jdx][right_var]) % modulus
            if right_delta == 0:
                continue
            coeff = (left_delta * pow(right_delta, -1, modulus)) % modulus
            bias = (models[idx][left_var] - coeff * models[idx][right_var]) % modulus
            candidate = (coeff, bias)
            break
        if candidate is not None:
            break
    if candidate is None:
        return None

    coeff, bias = candidate
    for model in models:
        if model[left_var] != (coeff * model[right_var] + bias) % modulus:
            return None
    return candidate


def _affine_rhs_expr(
    var_name: str, coeff: int, bias: int, modulus: int
) -> FieldExpr:
    terms: List[FieldExpr] = []
    coeff %= modulus
    bias %= modulus
    if coeff == 1:
        terms.append(FieldVar(var_name))
    elif coeff != 0:
        terms.append(FieldMul(FieldConst(coeff, modulus), FieldVar(var_name)))
    if bias != 0:
        terms.append(FieldConst(bias, modulus))
    if not terms:
        return FieldConst(0, modulus)
    if len(terms) == 1:
        return terms[0]
    return FieldAdd(*terms)
