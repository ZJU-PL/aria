"""Constrained candidate generation for concolic frontiers."""

from __future__ import annotations

from collections import OrderedDict
import copy
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import z3

from aria.sampling import (
    Logic,
    SamplingMethod,
    SamplingOptions,
    sample_models_from_formula,
)

from .models import Diagnostic, DiagnosticCode, DiagnosticSeverity
from .locking import z3_global_lock
from .symbolic import concrete_to_z3, decode_z3_string_escapes


@dataclass
class CandidateBatch:
    candidates: List[Dict[str, Any]]
    diagnostics: List[Diagnostic]
    stats: Dict[str, Any]


class ConstraintSampler:
    """Adapter from path-frontier formulas to Aria's sampler API."""

    def __init__(self, cache_size: int = 1_024) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.cache_size = cache_size
        self._cache: OrderedDict[str, CandidateBatch] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def sample(
        self,
        formula: z3.BoolRef,
        input_symbols: Mapping[str, z3.ExprRef],
        count: int,
        random_seed: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> CandidateBatch:
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when provided")
        with z3_global_lock():
            cache_key = self._cache_key(
                formula,
                input_symbols,
                count,
                random_seed,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._cache.move_to_end(cache_key)
                result = copy.deepcopy(cached)
                result.stats["cache_hit"] = True
                return result
            self._cache_misses += 1
            return self._sample_locked(
                formula,
                input_symbols,
                count,
                random_seed,
                timeout_seconds,
                cache_key,
            )

    def _sample_locked(
        self,
        formula: z3.BoolRef,
        input_symbols: Mapping[str, z3.ExprRef],
        count: int,
        random_seed: Optional[int],
        timeout_seconds: Optional[float],
        cache_key: str,
    ) -> CandidateBatch:
        diagnostics: List[Diagnostic] = []
        logic = _infer_logic(formula, input_symbols.values())
        try:
            result = sample_models_from_formula(
                formula,
                logic,
                SamplingOptions(
                    method=SamplingMethod.ENUMERATION,
                    num_samples=count,
                    random_seed=random_seed,
                    timeout=timeout_seconds,
                    projection_terms=list(input_symbols.values()),
                    tracked_terms=list(input_symbols.values()),
                    return_full_model=True,
                ),
            )
            candidates = [
                candidate
                for sample in result.samples
                if (candidate := _normalize_sample(sample, input_symbols)) is not None
                and _satisfies(formula, input_symbols, candidate)
            ]
            if len(candidates) >= count:
                batch = CandidateBatch(
                    candidates[:count], diagnostics, dict(result.stats)
                )
                self._remember(cache_key, batch)
                return batch
        # Sampler implementations currently expose heterogeneous exception types.
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.SAMPLER_FAILURE,
                    (
                        f"Aria sampler failed for {logic.value}; using exact Z3 "
                        f"enumeration: {exc}"
                    ),
                    DiagnosticSeverity.WARNING,
                )
            )
            candidates = []

        fallback, fallback_status = _enumerate_with_z3(
            formula,
            input_symbols,
            count,
            random_seed,
            timeout_seconds,
        )
        merged = _deduplicate([*locals().get("candidates", []), *fallback])[:count]
        batch = CandidateBatch(
            merged,
            diagnostics,
            {
                "method": "aria-with-z3-fallback",
                "candidate_count": len(merged),
                "solver_status": fallback_status,
                "cache_hit": False,
            },
        )
        if fallback_status in {"sat", "unsat"}:
            self._remember(cache_key, batch)
        return batch

    @property
    def cache_info(self) -> Dict[str, int]:
        return {
            "size": len(self._cache),
            "capacity": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def _remember(self, key: str, batch: CandidateBatch) -> None:
        self._cache[key] = copy.deepcopy(batch)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(
        formula: z3.BoolRef,
        symbols: Mapping[str, z3.ExprRef],
        count: int,
        random_seed: Optional[int],
    ) -> str:
        declarations = tuple(
            sorted((name, str(symbol.sort())) for name, symbol in symbols.items())
        )
        return repr((formula.sexpr(), declarations, count, random_seed))


def _infer_logic(formula: z3.ExprRef, symbols: Sequence[z3.ExprRef]) -> Logic:
    has_string = any(z3.is_string(symbol) for symbol in symbols)
    has_real = any(z3.is_real(symbol) for symbol in symbols)
    has_int = any(z3.is_int(symbol) for symbol in symbols)
    has_fp = any(z3.is_fp(symbol) for symbol in symbols)
    if has_fp:
        return Logic.QF_FP
    if has_string:
        return Logic.QF_SLIA
    if has_real and has_int:
        return Logic.QF_LIRA
    if has_real:
        return Logic.QF_NRA if _contains_nonlinear_arithmetic(formula) else Logic.QF_LRA
    if has_int:
        return Logic.QF_NIA if _contains_nonlinear_arithmetic(formula) else Logic.QF_LIA
    return Logic.QF_BOOL


def _contains_nonlinear_arithmetic(formula: z3.ExprRef) -> bool:
    stack = [formula]
    while stack:
        current = stack.pop()
        if z3.is_app(current) and current.decl().kind() in {
            z3.Z3_OP_MUL,
            z3.Z3_OP_POWER,
        }:
            if sum(not _is_numeric_literal(child) for child in current.children()) > 1:
                return True
        stack.extend(current.children())
    return False


def _is_numeric_literal(expression: z3.ExprRef) -> bool:
    return bool(
        z3.is_int_value(expression)
        or z3.is_rational_value(expression)
        or z3.is_algebraic_value(expression)
    )


def _normalize_sample(
    sample: Mapping[str, Any], symbols: Mapping[str, z3.ExprRef]
) -> Optional[Dict[str, Any]]:
    candidate: Dict[str, Any] = {}
    for name, symbol in symbols.items():
        if name in sample:
            value = sample[name]
        elif str(symbol) in sample:
            value = sample[str(symbol)]
        else:
            return None
        if z3.is_bool(symbol):
            candidate[name] = bool(value)
        elif z3.is_int(symbol):
            try:
                candidate[name] = int(value)
            except (TypeError, ValueError):
                return None
        elif z3.is_string(symbol):
            candidate[name] = decode_z3_string_escapes(str(value))
        elif z3.is_fp(symbol):
            converted = _sample_float(value)
            if converted is None:
                return None
            candidate[name] = converted
        else:
            return None
    return candidate


def _enumerate_with_z3(
    formula: z3.BoolRef,
    symbols: Mapping[str, z3.ExprRef],
    count: int,
    random_seed: Optional[int],
    timeout_seconds: Optional[float],
) -> tuple[List[Dict[str, Any]], str]:
    solver = z3.Solver()
    if random_seed is not None:
        solver.set(random_seed=random_seed)
    if timeout_seconds is not None:
        solver.set(timeout=max(1, int(timeout_seconds * 1_000)))
    solver.add(formula)
    candidates: List[Dict[str, Any]] = []
    status = "unknown"
    for _ in range(count):
        check_result = solver.check()
        status = str(check_result)
        if check_result != z3.sat:
            break
        model = solver.model()
        candidate: Dict[str, Any] = {}
        blockers = []
        for name, symbol in symbols.items():
            value = model.eval(symbol, model_completion=True)
            if z3.is_bool(symbol):
                python_value = z3.is_true(value)
            elif z3.is_int(symbol):
                python_value = value.as_long()
            elif z3.is_string(symbol):
                python_value = value.as_string()
            elif z3.is_fp(symbol):
                python_value = _z3_float(value)
            else:
                break
            candidate[name] = python_value
            blockers.append(symbol != value)
        else:
            candidates.append(candidate)
            if blockers:
                solver.add(z3.Or(*blockers))
                continue
        break
    return candidates, status


def _sample_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"bits=0x([0-9a-fA-F]+)", str(value))
    if match is None:
        return None
    bits = int(match.group(1), 16)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _z3_float(value: z3.ExprRef) -> float:
    bits = z3.simplify(z3.fpToIEEEBV(value))
    if not z3.is_bv_value(bits):
        raise ValueError(f"cannot convert floating-point model value {value}")
    return struct.unpack(">d", struct.pack(">Q", bits.as_long()))[0]


def _satisfies(
    formula: z3.BoolRef,
    symbols: Mapping[str, z3.ExprRef],
    candidate: Mapping[str, Any],
) -> bool:
    solver = z3.Solver()
    solver.add(formula)
    for name, symbol in symbols.items():
        concrete = concrete_to_z3(candidate[name])
        if concrete is None:
            return False
        solver.add(symbol == concrete)
    return solver.check() == z3.sat


def _deduplicate(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(sorted((name, repr(value)) for name, value in candidate.items()))
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result
