# sic_smt/solver.py -----------------------------------------------------------
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from z3 import (  # type: ignore
    ExprRef,
    QuantifierRef,
    And,
    Exists,
    ForAll,
    Or,
    ModelRef,
    Then,
    is_quantifier,
    simplify,
    substitute_vars,
    Solver,
    sat,
    unsat,
    unknown,
    AstVector,
    Z3_OP_AND,
    Z3_OP_OR,
    Z3_OP_ADD,
    Z3_OP_SUB,
    Z3_OP_MUL,
    Z3_OP_ITE,
)

from .taint import infer_sic_and_wic
from .util import fresh_const, project_model, let_sharing
from aria.utils.z3_expr_utils import get_variables
import logging

logger = logging.getLogger(__name__)


class QuantSolver:
    """
    One-shot quantifier solver based on taint-generated SICs.
    Works for any theory with decidable QF fragment (handled by Z3).
    """

    def __init__(
        self,
        *,
        timeout_ms: Optional[int] = None,
        confirm_unsat: bool = True,
        simplify_sic: bool = True,
        verify_wic: bool = False,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.confirm_unsat = confirm_unsat
        self.simplify_sic = simplify_sic
        self.verify_wic = verify_wic

    # ------------------------------------------------------------------ core
    def solve(self, formula: ExprRef) -> Tuple[str, Optional[ModelRef]]:
        """
        Return (result, model):

        * result ∈ {"sat","unsat","unknown"}
        * model  : model of the **original** formula (free symbols only)
        """
        # 0.  normalise   <--  NEW
        if isinstance(formula, AstVector):
            formula = And(*list(formula))
        elif isinstance(formula, (list, tuple)):
            formula = And(*formula)
        work = formula
        keep_vars = get_variables(work)

        # 1. eliminate quantifiers wherever they appear
        work, wic_so_far = self._eliminate_all_quantifiers(work)
        work = self._simplify_expr(work)

        # 2. residual formula is QF
        if self._contains_quantifier(work):
            return "unknown", None

        s = Solver()
        if self.timeout_ms:
            s.set("timeout", self.timeout_ms)
        s.add(work)
        r = s.check()
        if r == sat:
            return "sat", project_model(s.model(), keep_vars)
        if r == unsat:
            if wic_so_far:
                return "unsat", None
            return "unknown", None
        return "unknown", None

    # ---------------------------------------------------------------- helpers
    def _eliminate_all_quantifiers(
        self, expr: ExprRef, universal_depth: int = 0
    ) -> Tuple[ExprRef, bool]:
        """Recursively eliminate quantifiers while preserving dependencies."""
        if is_quantifier(expr):
            return self._eliminate_quantifier(expr, universal_depth)

        if expr.num_args() == 0:
            return expr, True

        rebuilt_children = []
        all_children_wic = True
        for c in expr.children():
            new_c, child_wic = self._eliminate_all_quantifiers(c, universal_depth)
            rebuilt_children.append(new_c)
            all_children_wic = all_children_wic and child_wic
        rebuilt = self._rebuild_expr(expr, rebuilt_children)
        return rebuilt, all_children_wic

    def _eliminate_quantifier(
        self, q: QuantifierRef, universal_depth: int
    ) -> Tuple[ExprRef, bool]:
        consts, body = self._open_quantifier(q)
        child_depth = universal_depth + 1 if q.is_forall() else universal_depth
        reduced_body, body_wic = self._eliminate_all_quantifiers(body, child_depth)
        reduced_body = self._simplify_expr(let_sharing(reduced_body))

        if q.is_forall():
            eliminated, is_wic = self._try_eliminate_forall(
                reduced_body, consts, universal_depth
            )
            if eliminated is not None:
                return eliminated, body_wic and is_wic
            return self._bind_quantifier(q, consts, reduced_body), False

        if universal_depth == 0:
            return reduced_body, body_wic
        return Exists(consts, reduced_body), body_wic

    def _try_eliminate_forall(
        self,
        body: ExprRef,
        targets: Sequence[ExprRef],
        outer_universal_depth: int,
    ) -> Tuple[Optional[ExprRef], bool]:
        matrix, witnesses = self._peel_existential_prefix(body)
        if self._contains_quantifier(matrix):
            return None, False
        matrix = self._simplify_expr(matrix)

        sic, is_wic = infer_sic_and_wic(
            matrix,
            set(targets),
            do_simplify=self.simplify_sic,
            verify_wic=self.verify_wic,
        )

        reduced = self._simplify_expr(And(matrix, sic))
        if outer_universal_depth > 0:
            bound = list(targets) + witnesses
            if bound:
                reduced = Exists(bound, reduced)
        return reduced, is_wic

    def _peel_existential_prefix(
        self, expr: ExprRef
    ) -> Tuple[ExprRef, List[ExprRef]]:
        witnesses: List[ExprRef] = []
        cur = expr
        while is_quantifier(cur) and cur.is_exists():
            consts, cur = self._open_quantifier(cur)
            witnesses.extend(consts)
        return cur, witnesses

    @staticmethod
    def _open_quantifier(q: QuantifierRef) -> Tuple[List[ExprRef], ExprRef]:
        consts = [
            fresh_const(q.var_sort(i), "q")
            for i in range(q.num_vars())
        ]
        body = q.body()
        if consts:
            body = substitute_vars(body, *reversed(consts))
        return consts, body

    @staticmethod
    def _bind_quantifier(
        q: QuantifierRef, consts: Sequence[ExprRef], body: ExprRef
    ) -> ExprRef:
        if q.is_forall():
            return ForAll(list(consts), body)
        return Exists(list(consts), body)

    @staticmethod
    def _simplify_expr(expr: ExprRef) -> ExprRef:
        return simplify(expr)

    @staticmethod
    def _rebuild_expr(expr: ExprRef, children: list[ExprRef]) -> ExprRef:
        kind = expr.decl().kind()
        if kind == Z3_OP_AND:
            return And(children)
        if kind == Z3_OP_OR:
            return Or(children)
        if kind in (Z3_OP_ADD, Z3_OP_SUB, Z3_OP_MUL):
            return expr.decl()(*children)
        if kind == Z3_OP_ITE:
            return expr.decl()(*children)
        if expr.num_args() == len(children):
            return expr.decl()(*children)
        # Fallback: keep original if rebuilding fails
        try:
            return expr.decl()(*children)
        except Exception:
            return expr

    @staticmethod
    def _contains_quantifier(expr: ExprRef) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if is_quantifier(node):
                return True
            stack.extend(node.children())
        return False

    # ------------------------------------------------------------------ file


def solve_file(path: str, *, timeout_ms: Optional[int] = None) -> None:
    from z3 import parse_smt2_file  # late import

    f = parse_smt2_file(path)
    if isinstance(f, list):
        f = And(*f)
    solver = QuantSolver(timeout_ms=timeout_ms)
    res, model = solver.solve(f)
    print(res)
    if model is not None:
        print(model)
