"""Solver-guided shrinking of reproducible concolic failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, TYPE_CHECKING

import z3

from .locking import z3_global_lock
from .models import Diagnostic, DiagnosticCode, ExecutionOutcome, ExecutionTrace
from .structured import materialize_candidate

if TYPE_CHECKING:
    from .engine import NativeConcolicEngine


@dataclass
class ShrinkResult:
    """The smallest validated counterexample found by one shrink pass."""

    original: ExecutionTrace
    shrunk: ExecutionTrace
    changed: bool
    solver_status: str
    diagnostics: List[Diagnostic]


class CounterexampleShrinker:
    """Minimize primitive inputs while preserving the observed failing path."""

    def __init__(self, engine: "NativeConcolicEngine") -> None:
        self.engine = engine

    def shrink(self, trace: ExecutionTrace) -> ShrinkResult:
        candidate, status, early = self._optimized_candidate(trace)
        if early is not None:
            return early
        assert candidate is not None
        shrunk = self.engine.trace(candidate)
        return self._validated_result(trace, shrunk, candidate, status)

    async def ashrink(self, trace: ExecutionTrace) -> ShrinkResult:
        """Minimize and revalidate a failure produced by an async target."""
        candidate, status, early = self._optimized_candidate(trace)
        if early is not None:
            return early
        assert candidate is not None
        shrunk = await self.engine.atrace(candidate)
        return self._validated_result(trace, shrunk, candidate, status)

    def _optimized_candidate(self, trace: ExecutionTrace):
        if trace.outcome != ExecutionOutcome.RAISED:
            raise ValueError("only failing traces can be shrunk")
        if not trace.input_symbols:
            return (
                None,
                "no-symbols",
                ShrinkResult(trace, trace, False, "no-symbols", []),
            )
        path_formula = self._path_formula(trace)
        with z3_global_lock():
            optimizer = z3.Optimize()
            optimizer.set(priority="lex")
            optimizer.add(path_formula)
            for symbol in trace.input_symbols.values():
                if z3.is_bool(symbol):
                    optimizer.minimize(z3.If(symbol, 1, 0))
                elif z3.is_int(symbol):
                    optimizer.minimize(z3.If(symbol >= 0, symbol, -symbol))
                    optimizer.minimize(symbol)
                elif z3.is_string(symbol):
                    optimizer.minimize(z3.Length(symbol))
            status = optimizer.check()
            if status != z3.sat:
                return (
                    None,
                    str(status),
                    ShrinkResult(trace, trace, False, str(status), []),
                )
            flat_candidate = self._candidate(optimizer.model(), trace.input_symbols)
            candidate = materialize_candidate(
                trace.inputs,
                flat_candidate,
                trace.input_paths,
            )
        return candidate, str(status), None

    @staticmethod
    def _validated_result(
        trace: ExecutionTrace,
        shrunk: ExecutionTrace,
        candidate: Dict[str, Any],
        status: str,
    ) -> ShrinkResult:
        if not _same_failure(trace, shrunk):
            diagnostic = Diagnostic(
                DiagnosticCode.INVALID_SAMPLE,
                (
                    "optimized input did not reproduce the original failure; "
                    "keeping the original"
                ),
            )
            return ShrinkResult(trace, trace, False, "not-reproduced", [diagnostic])
        return ShrinkResult(
            trace,
            shrunk,
            candidate != trace.inputs,
            status,
            [],
        )

    @staticmethod
    def _path_formula(trace: ExecutionTrace) -> z3.BoolRef:
        constraints = []
        for branch in trace.branches:
            if branch.predicate is None:
                continue
            constraints.extend(branch.guards)
            constraints.append(
                branch.predicate if branch.taken else z3.Not(branch.predicate)
            )
        return z3.And(*constraints)

    @staticmethod
    def _candidate(
        model: z3.ModelRef, symbols: Mapping[str, z3.ExprRef]
    ) -> Dict[str, Any]:
        candidate: Dict[str, Any] = {}
        for name, symbol in symbols.items():
            value = model.eval(symbol, model_completion=True)
            if z3.is_bool(symbol):
                candidate[name] = z3.is_true(value)
            elif z3.is_int(symbol):
                candidate[name] = value.as_long()
            elif z3.is_string(symbol):
                candidate[name] = value.as_string()
            else:
                raise ValueError(f"cannot shrink input sort {symbol.sort()}")
        return candidate


def _same_failure(original: ExecutionTrace, candidate: ExecutionTrace) -> bool:
    return (
        candidate.outcome == ExecutionOutcome.RAISED
        and candidate.exception_type == original.exception_type
        and candidate.exception_message == original.exception_message
    )
