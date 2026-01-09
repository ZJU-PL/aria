# sic_smt/solver.py -----------------------------------------------------------
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from z3 import (  # type: ignore
    ExprRef,
    QuantifierRef,
    And,
    Or,
    ModelRef,
    is_quantifier,
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
from .util import skolemize_block, project_model
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
    ) -> None:
        self.timeout_ms = timeout_ms
        self.confirm_unsat = confirm_unsat
        self.simplify_sic = simplify_sic

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

        wic_so_far = True
        # 1. eliminate quantifiers wherever they appear
        work, wic_so_far = self._eliminate_all_quantifiers(work, wic_so_far)

        # 2. residual formula is QF
        if self._contains_quantifier(work):
            r, model = self._solve_quantified(work)
            return r, project_model(model, keep_vars) if model is not None else None

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
            if self.confirm_unsat and self._check_quantified_unsat(formula):
                return "unsat", None
            return "unknown", None
        return "unknown", None

    # ---------------------------------------------------------------- helpers
    def _eliminate_head_block(
        self, formula: ExprRef, q: QuantifierRef
    ) -> Tuple[ExprRef, bool]:
        """Return formula where the head quantifier block is removed."""
        assert is_quantifier(formula)
        assert q.eq(formula)

        sk_body, skol_consts = skolemize_block(q)
        if q.is_forall():
            targets = set(skol_consts.values())
            sic, is_wic = infer_sic_and_wic(
                sk_body, targets, do_simplify=self.simplify_sic
            )
            return And(sk_body, sic), is_wic
        return sk_body, True

    def _eliminate_all_quantifiers(
        self, expr: ExprRef, wic_acc: bool
    ) -> Tuple[ExprRef, bool]:
        """Recursively eliminate all quantifiers in expr."""
        if is_quantifier(expr):
            new_expr, is_wic = self._eliminate_head_block(expr, expr)
            return self._eliminate_all_quantifiers(new_expr, wic_acc and is_wic)

        if expr.num_args() == 0:
            return expr, wic_acc

        rebuilt_children = []
        child_wic = wic_acc
        for c in expr.children():
            new_c, child_wic = self._eliminate_all_quantifiers(c, child_wic)
            rebuilt_children.append(new_c)
        rebuilt = self._rebuild_expr(expr, rebuilt_children)
        return rebuilt, child_wic

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

    def _check_quantified_unsat(self, formula: ExprRef) -> bool:
        """Confirm unsat result with Z3's quantified reasoning."""
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(formula)
        return solver.check() == unsat

    @staticmethod
    def _contains_quantifier(expr: ExprRef) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if is_quantifier(node):
                return True
            stack.extend(node.children())
        return False

    def _solve_quantified(self, formula: ExprRef) -> Tuple[str, Optional[ModelRef]]:
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(formula)
        res = solver.check()
        if res == sat:
            return "sat", solver.model()
        if res == unsat:
            return "unsat", None
        return "unknown", None

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
