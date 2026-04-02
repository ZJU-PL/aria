# sic_smt/solver.py -----------------------------------------------------------
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, cast
from z3 import (  # type: ignore
    ExprRef,
    QuantifierRef,
    And,
    Exists,
    ForAll,
    ModelRef,
    is_quantifier,
    simplify,
    substitute_vars,
    substitute,
    Solver,
    sat,
    unsat,
    AstVector,
    Not,
)

from .taint import infer_sic_and_wic
from .util import fresh_const, project_model
from aria.utils.z3_expr_utils import get_variables
import logging

logger = logging.getLogger(__name__)


class QuantSolver:
    """
    One-shot solver for prenex exists-forall formulas with a quantifier-free
    matrix, based on taint-generated SICs.

    Supported fragment:
        exists X . forall Y . P(X, Y)

    Free variables are treated as existentially quantified parameters. Any
    nested or alternating quantifier structure outside this fragment returns
    ``unknown``.
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
        if isinstance(formula, AstVector):
            items = [formula[i] for i in range(len(formula))]
            formula = (
                cast(ExprRef, items[0])
                if len(items) == 1
                else cast(ExprRef, And(*items))
            )
        elif isinstance(formula, (list, tuple)):
            formula = (
                cast(ExprRef, formula[0])
                if len(formula) == 1
                else cast(ExprRef, And(*formula))
            )
        work = formula
        keep_vars = get_variables(work)

        parsed = self._parse_exists_forall_prefix(work)
        if parsed is None:
            return "unknown", None
        matrix, universal_consts = parsed

        if not universal_consts:
            return self._solve_qf(matrix, keep_vars)

        sic, _ = infer_sic_and_wic(
            matrix,
            set(universal_consts),
            do_simplify=self.simplify_sic,
            verify_wic=self.verify_wic,
        )
        reduced = self._simplify_expr(cast(ExprRef, And(matrix, sic)))

        if not self._check_sound_reduction(matrix, universal_consts, reduced):
            return "unknown", None

        result, model = self._solve_qf(reduced, keep_vars)
        if result == "sat":
            return result, model
        if result == "unknown":
            return result, model

        if self._check_complete_reduction(matrix, universal_consts, sic):
            return "unsat", None
        return "unknown", None

    # ---------------------------------------------------------------- helpers
    def _solve_qf(
        self, formula: ExprRef, keep_vars: Sequence[ExprRef]
    ) -> Tuple[str, Optional[ModelRef]]:
        if self._contains_quantifier(formula):
            return "unknown", None

        s = Solver()
        if self.timeout_ms:
            s.set("timeout", self.timeout_ms)
        s.add(formula)
        r = s.check()
        if r == sat:
            return "sat", project_model(s.model(), keep_vars)
        if r == unsat:
            return "unsat", None
        return "unknown", None

    def _parse_exists_forall_prefix(
        self, expr: ExprRef
    ) -> Optional[Tuple[ExprRef, List[ExprRef]]]:
        universal_consts: List[ExprRef] = []
        cur = expr

        while is_quantifier(cur) and cast(QuantifierRef, cur).is_exists():
            consts, cur = self._open_quantifier(cast(QuantifierRef, cur))

        while is_quantifier(cur) and cast(QuantifierRef, cur).is_forall():
            consts, cur = self._open_quantifier(cast(QuantifierRef, cur))
            universal_consts.extend(consts)

        if is_quantifier(cur):
            return None

        cur = self._simplify_expr(cur)
        if self._contains_quantifier(cur):
            return None
        return cur, universal_consts

    def _check_sound_reduction(
        self,
        matrix: ExprRef,
        targets: Sequence[ExprRef],
        reduced: ExprRef,
    ) -> bool:
        other_targets = [fresh_const(t.sort(), "y_sound") for t in targets]
        other_matrix = substitute(matrix, *list(zip(targets, other_targets)))
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(reduced, Not(other_matrix))
        return solver.check() == unsat

    def _check_complete_reduction(
        self,
        matrix: ExprRef,
        targets: Sequence[ExprRef],
        sic: ExprRef,
    ) -> bool:
        other_targets = [fresh_const(t.sort(), "y_complete") for t in targets]
        universally_valid_matrix = ForAll(
            list(other_targets),
            substitute(matrix, *list(zip(targets, other_targets))),
        )
        solver = Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)
        solver.add(universally_valid_matrix, Not(sic))
        return solver.check() == unsat

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
    def _simplify_expr(expr: ExprRef) -> ExprRef:
        return cast(ExprRef, simplify(expr))

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
        f = cast(ExprRef, And(*f))
    solver = QuantSolver(timeout_ms=timeout_ms)
    res, model = solver.solve(cast(ExprRef, f))
    print(res)
    if model is not None:
        print(model)
