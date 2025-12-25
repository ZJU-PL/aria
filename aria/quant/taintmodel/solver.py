# sic_smt/solver.py -----------------------------------------------------------
from __future__ import annotations
import itertools
from typing import Iterable, List, Sequence, Tuple
from z3 import (  # type: ignore
    ExprRef,
    BoolRef,
    BoolVal,
    Const,
    ForAll,
    Exists,
    QuantifierRef,
    And,
    Or,
    ModelRef,
    is_quantifier,
    is_false,
    is_true,
    substitute,
    Not,
    Solver,
    sat,
    unsat,
    unknown,
    AstVector,
)

from .taint import infer_sic
from .util import fresh_const, iter_qblocks, skolemize_block, project_model


class QuantSolver:
    """
    One-shot quantifier solver based on taint-generated SICs.
    Works for any theory with decidable QF fragment (handled by Z3).
    """

    def __init__(self, *, timeout_ms: int | None = None) -> None:
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------ core
    def solve(self, formula: ExprRef) -> Tuple[str, ModelRef | None]:
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

        # 1. iterate over quantifier blocks, *outermost first*
        for q in iter_qblocks(work):
            work = self._eliminate_block(work, q)

        # 2. residual formula is QF
        s = Solver()
        if self.timeout_ms:
            s.set("timeout", self.timeout_ms)
        s.add(work)
        r = s.check()
        if r == sat:
            return "sat", s.model()
        if r == unsat:
            return "unsat", None
        return "unknown", None

    # ---------------------------------------------------------------- helpers
    def _eliminate_block(self, formula: ExprRef, q: QuantifierRef) -> ExprRef:
        """Return formula where *q* is removed via SIC + Skolem."""
        assert is_quantifier(q)
        body = q.body()

        # 1. Skolemise *one* block (∀ or ∃)
        sk_body, skol_consts = skolemize_block(q)

        # 2. Decide which of the fresh constants are *targets*
        #    –∀  : targets  – we need independence from them
        #    –∃  : they become *free* parameters ➜ keep dependence
        targets = (
            set(skol_consts.values()) if q.is_forall() else set()
        )  # type: ignore[arg-type]

        # 3. Build the SIC (quantifier-free, same theory)
        sic = infer_sic(sk_body, targets) if targets else BoolVal(True)

        # 4. Conjoin and replace q in parent formula
        guarded = And(sk_body, sic)
        return substitute(formula, (q, guarded))

    # ------------------------------------------------------------------ file
def solve_file(path: str, *, timeout_ms: int | None = None) -> None:
    from z3 import parse_smt2_file  # late import

    f = parse_smt2_file(path)
    if isinstance(f, list):
        f = And(*f)
    solver = QuantSolver(timeout_ms=timeout_ms)
    res, model = solver.solve(f)
    print(res)
    if model is not None:
        print(model)
