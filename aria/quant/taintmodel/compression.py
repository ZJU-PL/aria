"""Certified, low-dimensional representations of universal counterexamples.

The heuristic code in this module is deliberately untrusted.  A proposed
compression is useful only after :func:`certify_compression` proves the
whole-matrix counterexample-preservation obligation with a quantifier-free
query.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

from z3 import (  # type: ignore
    And,
    BitVecVal,
    BoolRef,
    BoolVal,
    ExprRef,
    ModelRef,
    Not,
    Solver,
    ZeroExt,
    is_and,
    is_app,
    is_bool,
    is_distinct,
    is_eq,
    is_not,
    is_or,
    is_quantifier,
    sat,
    simplify,
    substitute,
    unsat,
    Z3_BV_SORT,
    Z3_BOOL_SORT,
    Z3_OP_BAND,
    Z3_OP_EXTRACT,
)

from aria.utils.z3.expr import get_variables

from .util import fresh_const


def _contains_quantifier(expr: ExprRef) -> bool:
    stack = [expr]
    while stack:
        node = stack.pop()
        if is_quantifier(node):
            return True
        stack.extend(node.children())
    return False


def _is_member(symbol: ExprRef, symbols: Sequence[ExprRef]) -> bool:
    return any(symbol.eq(other) for other in symbols)


def _members(expr: ExprRef, symbols: Sequence[ExprRef]) -> List[ExprRef]:
    return [
        symbol
        for symbol in symbols
        if any(symbol.eq(variable) for variable in get_variables(expr))
    ]


def _variables_are_within(expr: ExprRef, allowed: Sequence[ExprRef]) -> bool:
    return all(_is_member(variable, allowed) for variable in get_variables(expr))


def _bit_width(expr: ExprRef) -> Optional[int]:
    kind = expr.sort().kind()
    if kind == Z3_BOOL_SORT:
        return 1
    if kind == Z3_BV_SORT:
        return expr.size()
    return None


@dataclass(frozen=True)
class CounterexampleCompression:
    """An explicit guarded counterexample representation ``(G, tau, pi)``.

    ``reconstruction`` is ``tau(X, Z)`` in the order of ``universals`` and
    ``projection`` is ``pi(X, Y)`` in the order of ``residuals``.
    """

    guard: BoolRef
    universals: Tuple[ExprRef, ...]
    residuals: Tuple[ExprRef, ...]
    reconstruction: Tuple[ExprRef, ...]
    projection: Tuple[ExprRef, ...]
    name: str = "compression"

    @classmethod
    def identity(
        cls,
        universals: Sequence[ExprRef],
    ) -> "CounterexampleCompression":
        residuals = tuple(fresh_const(var.sort(), "z_identity") for var in universals)
        return cls(
            guard=BoolVal(True),
            universals=tuple(universals),
            residuals=residuals,
            reconstruction=residuals,
            projection=tuple(universals),
            name="identity",
        )

    @property
    def residual_bit_count(self) -> Optional[int]:
        widths = [_bit_width(residual) for residual in self.residuals]
        if any(width is None for width in widths):
            return None
        return sum(cast(int, width) for width in widths)

    def validate(self, existentials: Sequence[ExprRef]) -> None:
        """Reject ill-sorted terms or hidden free variables.

        This structural check is separate from semantic certification.  In
        particular, it prevents an eliminated universal from being smuggled
        into ``tau`` as a free constant.
        """
        if not is_bool(self.guard):
            raise ValueError("compression guard must be Boolean")
        if len(self.reconstruction) != len(self.universals):
            raise ValueError("tau must reconstruct every universal")
        if len(self.projection) != len(self.residuals):
            raise ValueError("pi must provide every residual choice")
        if _contains_quantifier(self.guard):
            raise ValueError("compression guard must be quantifier-free")
        if not _variables_are_within(self.guard, existentials):
            raise ValueError("compression guard may mention only existentials")

        reconstruction_scope = tuple(existentials) + self.residuals
        projection_scope = tuple(existentials) + self.universals
        for universal, term in zip(self.universals, self.reconstruction):
            if not universal.sort().eq(term.sort()):
                raise ValueError("reconstruction term has the wrong sort")
            if _contains_quantifier(term):
                raise ValueError("reconstruction terms must be quantifier-free")
            if not _variables_are_within(term, reconstruction_scope):
                raise ValueError("reconstruction contains a hidden dependency")
        for residual, term in zip(self.residuals, self.projection):
            if not residual.sort().eq(term.sort()):
                raise ValueError("projection term has the wrong sort")
            if _contains_quantifier(term):
                raise ValueError("projection terms must be quantifier-free")
            if not _variables_are_within(term, projection_scope):
                raise ValueError("projection contains a hidden dependency")

    def compressed_matrix(self, matrix: ExprRef) -> ExprRef:
        return cast(
            ExprRef,
            substitute(matrix, *list(zip(self.universals, self.reconstruction))),
        )

    def projected_reconstruction(self) -> Tuple[ExprRef, ...]:
        substitutions = list(zip(self.residuals, self.projection))
        return tuple(
            cast(ExprRef, substitute(term, *substitutions))
            for term in self.reconstruction
        )

    def certificate_obligation(self, matrix: ExprRef) -> BoolRef:
        reconstructed = substitute(
            matrix,
            *list(zip(self.universals, self.projected_reconstruction())),
        )
        return cast(BoolRef, And(self.guard, Not(matrix), reconstructed))


def certify_compression(
    matrix: ExprRef,
    existentials: Sequence[ExprRef],
    compression: CounterexampleCompression,
    *,
    timeout_ms: Optional[int] = None,
) -> Tuple[str, Optional[ModelRef]]:
    """Check the whole-formula counterexample-preservation certificate.

    Returns ``("valid", None)``, ``("counterexample", model)``, or
    ``("unknown", None)``.  The only backend assertion is quantifier-free.
    """
    compression.validate(existentials)
    obligation = compression.certificate_obligation(matrix)
    if _contains_quantifier(obligation):
        raise ValueError("compression certificate is not quantifier-free")

    solver = Solver()
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    solver.add(obligation)
    result = solver.check()
    if result == unsat:
        return "valid", None
    if result == sat:
        return "counterexample", solver.model()
    return "unknown", None


def _flatten_or(expr: ExprRef) -> List[ExprRef]:
    if not is_or(expr):
        return [expr]
    result: List[ExprRef] = []
    for child in expr.children():
        result.extend(_flatten_or(child))
    return result


def _failure_conjuncts(matrix: ExprRef) -> List[ExprRef]:
    """Expose literals of ``not matrix`` without rewriting BV structure."""
    disjuncts = _flatten_or(matrix)
    conjuncts: List[ExprRef] = []
    for disjunct in disjuncts:
        if is_not(disjunct):
            conjuncts.append(disjunct.arg(0))
        elif is_distinct(disjunct) and disjunct.num_args() == 2:
            # Z3Py represents ``a != b`` as a binary Distinct. Construct the
            # negated equality directly so BV masks are not destroyed by a
            # general simplification pass.
            conjuncts.append(disjunct.arg(0) == disjunct.arg(1))
        else:
            conjuncts.append(cast(ExprRef, Not(disjunct)))
    return conjuncts


def _flatten_and(expr: ExprRef) -> List[ExprRef]:
    if not is_and(expr):
        return [expr]
    result: List[ExprRef] = []
    for child in expr.children():
        result.extend(_flatten_and(child))
    return result


def _band_parts(expr: ExprRef) -> List[ExprRef]:
    if not is_app(expr) or expr.decl().kind() != Z3_OP_BAND:
        return [expr]
    result: List[ExprRef] = []
    for child in expr.children():
        result.extend(_band_parts(child))
    return result


def _bv_and(parts: Sequence[ExprRef]) -> ExprRef:
    assert parts
    return reduce(lambda left, right: left & right, parts)


def _bv_or(parts: Sequence[ExprRef]) -> ExprRef:
    assert parts
    return reduce(lambda left, right: left | right, parts)


def _masked_occurrence(
    expr: ExprRef,
    universal: ExprRef,
) -> Optional[ExprRef]:
    parts = _band_parts(expr)
    positions = [index for index, part in enumerate(parts) if part.eq(universal)]
    if len(positions) != 1 or len(parts) < 2:
        return None
    mask_parts = [part for index, part in enumerate(parts) if index != positions[0]]
    mask = _bv_and(mask_parts)
    if _is_member(universal, get_variables(mask)):
        return None
    return mask


def _masked_piece_from_equality(
    equality: ExprRef,
    universal: ExprRef,
) -> Optional[Tuple[ExprRef, ExprRef]]:
    if not is_eq(equality) or universal.sort().kind() != Z3_BV_SORT:
        return None
    lhs, rhs = equality.children()
    for side, other in ((lhs, rhs), (rhs, lhs)):
        mask = _masked_occurrence(side, universal)
        if mask is None or _is_member(universal, get_variables(other)):
            continue
        return mask, cast(ExprRef, other & mask)
    return None


def _slice_piece_from_equality(
    equality: ExprRef,
    universal: ExprRef,
) -> Optional[Tuple[ExprRef, ExprRef]]:
    if not is_eq(equality) or universal.sort().kind() != Z3_BV_SORT:
        return None
    lhs, rhs = equality.children()
    for side, other in ((lhs, rhs), (rhs, lhs)):
        if not is_app(side) or side.decl().kind() != Z3_OP_EXTRACT:
            continue
        if side.num_args() != 1 or not side.arg(0).eq(universal):
            continue
        if _is_member(universal, get_variables(other)):
            continue
        high, low = side.decl().params()
        slice_width = high - low + 1
        if other.sort().kind() != Z3_BV_SORT or other.size() != slice_width:
            continue
        width = universal.size()
        mask_value = ((1 << slice_width) - 1) << low
        mask = BitVecVal(mask_value, width)
        extended = ZeroExt(width - slice_width, other)
        piece = extended if low == 0 else extended << low
        return mask, cast(ExprRef, piece & mask)
    return None


def _is_unconditional_cover(
    masks: Sequence[ExprRef],
    *,
    timeout_ms: Optional[int],
) -> bool:
    if not masks:
        return False
    combined = _bv_or(masks)
    all_ones = BitVecVal((1 << combined.size()) - 1, combined.size())
    solver = Solver()
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    solver.add(combined != all_ones)
    return solver.check() == unsat


def _candidate_dependencies(
    term: ExprRef,
    universals: Sequence[ExprRef],
) -> List[ExprRef]:
    return _members(term, universals)


def _find_cycle(
    candidates: Dict[ExprRef, ExprRef],
    universals: Sequence[ExprRef],
) -> Optional[List[ExprRef]]:
    visiting: Set[ExprRef] = set()
    visited: Set[ExprRef] = set()
    path: List[ExprRef] = []

    def visit(node: ExprRef) -> Optional[List[ExprRef]]:
        if node in visiting:
            start = next(i for i, item in enumerate(path) if item.eq(node))
            return path[start:]
        if node in visited:
            return None
        visiting.add(node)
        path.append(node)
        for dependency in _candidate_dependencies(candidates[node], universals):
            if dependency in candidates:
                cycle = visit(dependency)
                if cycle is not None:
                    return cycle
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for universal in universals:
        if universal not in candidates:
            continue
        cycle = visit(universal)
        if cycle is not None:
            return cycle
    return None


def propose_joint_compression(
    matrix: ExprRef,
    existentials: Sequence[ExprRef],
    universals: Sequence[ExprRef],
    *,
    timeout_ms: Optional[int] = None,
) -> Optional[CounterexampleCompression]:
    """Propose a global failure-conditioned reconstruction.

    The initial bounded analysis combines direct Boolean/BV equalities and
    masked or extracted BV equalities across all conjuncts of a disjunctive
    failure condition.  Cyclic reconstructions are broken by retaining one
    universal as a residual choice.  Certification remains mandatory.
    """
    del existentials  # Reserved for guarded proposals in the next analysis stage.
    if not universals:
        return None
    if any(_bit_width(universal) is None for universal in universals):
        return None

    conjuncts: List[ExprRef] = []
    for conjunct in _failure_conjuncts(matrix):
        conjuncts.extend(_flatten_and(conjunct))

    candidates: Dict[ExprRef, ExprRef] = {}
    masked: Dict[ExprRef, List[Tuple[ExprRef, ExprRef]]] = {
        universal: [] for universal in universals
    }

    for conjunct in conjuncts:
        for universal in universals:
            # Boolean failure literals directly determine their value.
            if universal.sort().kind() == Z3_BOOL_SORT:
                if conjunct.eq(universal):
                    candidates[universal] = BoolVal(True)
                    continue
                if is_not(conjunct) and conjunct.arg(0).eq(universal):
                    candidates[universal] = BoolVal(False)
                    continue

            if is_eq(conjunct):
                lhs, rhs = conjunct.children()
                for side, other in ((lhs, rhs), (rhs, lhs)):
                    if side.eq(universal) and not _is_member(
                        universal, get_variables(other)
                    ):
                        candidates.setdefault(universal, other)

                piece = _masked_piece_from_equality(conjunct, universal)
                if piece is not None:
                    masked[universal].append(piece)
                slice_piece = _slice_piece_from_equality(conjunct, universal)
                if slice_piece is not None:
                    masked[universal].append(slice_piece)

    for universal, pieces in masked.items():
        if universal in candidates or not pieces:
            continue
        masks = [mask for mask, _ in pieces]
        if _is_unconditional_cover(masks, timeout_ms=timeout_ms):
            candidates[universal] = simplify(_bv_or([piece for _, piece in pieces]))

    # A dependency cycle is legal only if at least one member remains in Z.
    while True:
        cycle = _find_cycle(candidates, universals)
        if cycle is None:
            break
        residual = min(cycle, key=lambda item: item.sexpr())
        del candidates[residual]

    if not candidates:
        return None

    residual_universals = [
        universal for universal in universals if universal not in candidates
    ]
    residuals = tuple(
        fresh_const(universal.sort(), "z_residual") for universal in residual_universals
    )
    residual_map = dict(zip(residual_universals, residuals))
    expanded: Dict[ExprRef, ExprRef] = {}

    def expand(universal: ExprRef) -> ExprRef:
        if universal in expanded:
            return expanded[universal]
        if universal in residual_map:
            expanded[universal] = residual_map[universal]
            return expanded[universal]
        term = candidates[universal]
        substitutions = [
            (dependency, expand(dependency))
            for dependency in _candidate_dependencies(term, universals)
        ]
        expanded[universal] = cast(ExprRef, substitute(term, *substitutions))
        return expanded[universal]

    reconstruction = tuple(expand(universal) for universal in universals)
    proposal = CounterexampleCompression(
        guard=BoolVal(True),
        universals=tuple(universals),
        residuals=residuals,
        reconstruction=reconstruction,
        projection=tuple(residual_universals),
        name="joint-failure-reconstruction",
    )

    identity_bits = sum(cast(int, _bit_width(universal)) for universal in universals)
    if (
        proposal.residual_bit_count is None
        or proposal.residual_bit_count >= identity_bits
    ):
        return None
    return proposal
