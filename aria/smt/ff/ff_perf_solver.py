#!/usr/bin/env python3
"""Performance-oriented finite-field solver backend.

This backend combines adaptive modulo-reduction scheduling (ARS) with
prime-structured modular kernels (PSK). It uses integer translation and falls
back to stricter schedules when needed.
"""
from __future__ import annotations

import functools
import os
from typing import Dict, List, Optional

import z3

from .ff_ast import (
    BoolAnd,
    BoolConst,
    BoolIte,
    BoolImplies,
    BoolNot,
    BoolOr,
    BoolVar,
    BoolXor,
    FieldAdd,
    FieldConst,
    FieldDiv,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldSub,
    FieldVar,
    ParsedFormula,
    field_modulus_from_sort,
    infer_expr_sort,
    infer_field_modulus,
    is_bool_sort,
)
from .ff_int_solver import FFIntSolver
from .ff_ir import FFIRMetadata, build_ir_metadata, expr_key
from .ff_modkernels import ModKernelSelector, ModReducer
from .ff_numbertheory import is_probable_prime
from .ff_preprocess import preprocess_formula
from .ff_reduction_scheduler import ReductionScheduler, stricter_schedule


class FFPerfSolver:
    """Finite-field solver using ARS + PSK over integer constraints.

    The solver translates QF_FF formulas to integer arithmetic while:
    - delaying modulo operations with a schedule-aware policy;
    - selecting modulus-specific reduction kernels;
    - escalating schedules on ``unknown`` when recovery is enabled;
    - falling back to ``FFIntSolver`` as a safety net.
    """

    def __init__(
        self,
        schedule: str = "balanced",
        kernel_mode: str = "auto",
        recovery: bool = True,
    ):
        env_schedule = os.getenv("ARIA_FF_SCHEDULE")
        env_kernel = os.getenv("ARIA_FF_KERNEL_MODE")
        if env_schedule:
            schedule = env_schedule.strip()
        if env_kernel:
            kernel_mode = env_kernel.strip()

        self.initial_schedule = schedule
        self.kernel_mode = kernel_mode
        self.recovery = recovery
        self.solver = z3.SolverFor("QF_NIA")
        self.vars: Dict[str, z3.ExprRef] = {}
        self.var_sorts: Dict[str, str] = {}
        self._metadata: Optional[FFIRMetadata] = None
        self._reducer: Optional[ModReducer] = None
        self._stats: Dict[str, int] = {}

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Check satisfiability with adaptive schedule fallback."""
        normalized = preprocess_formula(formula)
        self._setup_fields(normalized.field_sizes)
        self.var_sorts = dict(normalized.variables)
        self._metadata = build_ir_metadata(normalized.assertions)

        schedules = self._schedule_chain(self.initial_schedule)
        last_result = z3.unknown
        for attempt, schedule in enumerate(schedules):
            self._reset_stats()
            self._stats["attempt"] = attempt + 1
            self._stats["fallback_attempts"] = attempt
            self._stats["schedule_%s" % schedule] = 1

            self._reset_solver()
            self._declare_vars(normalized.variables)
            self._setup_kernels(normalized.field_sizes)

            for assertion in normalized.assertions:
                self.solver.add(self._tr(assertion, schedule=schedule))

            result = self.solver.check()
            last_result = result
            if result in (z3.sat, z3.unsat):
                self._stats["final_schedule_%s" % schedule] = 1
                return result
            if not self.recovery:
                return result

        # Final safety fallback to the stable integer backend.
        fallback_solver = FFIntSolver()
        fallback_res = fallback_solver.check(normalized)
        self._stats["used_stable_fallback"] = 1
        if fallback_res in (z3.sat, z3.unsat):
            self.solver = fallback_solver.solver
            return fallback_res
        return last_result

    def model(self) -> Optional[z3.ModelRef]:
        """Return model when SAT and available."""
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    def stats(self) -> Dict[str, int]:
        """Return counters from the latest run."""
        return dict(self._stats)

    def _schedule_chain(self, schedule: str) -> List[str]:
        """Return schedule escalation order for one check call."""
        chain = [schedule]
        if not self.recovery:
            return chain
        next_schedule = stricter_schedule(schedule)
        while next_schedule is not None:
            chain.append(next_schedule)
            next_schedule = stricter_schedule(next_schedule)
        return chain

    def _reset_solver(self) -> None:
        self.solver = z3.SolverFor("QF_NIA")
        self.vars = {}

    def _reset_stats(self) -> None:
        self._stats = {
            "reductions_total": 0,
            "reduction_boundaries": 0,
            "reduction_nonboundary": 0,
            "kernel_generic": 0,
            "kernel_pseudo_mersenne": 0,
            "kernel_near_power2_sparse": 0,
            "kernel_small_prime_unrolled": 0,
        }

    def _setup_fields(self, fields) -> None:
        for modulus in fields:
            if not is_probable_prime(modulus):
                raise ValueError("Finite-field sort requires prime p, got %d" % modulus)

    def _setup_kernels(self, fields) -> None:
        """Classify all field moduli and initialize the reducer."""
        selector = ModKernelSelector(kernel_mode=self.kernel_mode)
        specs = {}
        for modulus in fields:
            spec = selector.classify(modulus)
            specs[modulus] = spec
            self._stats["kernel_%s" % spec.kind] = self._stats.get(
                "kernel_%s" % spec.kind, 0
            ) + 1
        self._reducer = ModReducer(specs)

    def _declare_vars(self, varmap: Dict[str, str]) -> None:
        """Declare z3 variables and range constraints for field elements."""
        for name, sort_id in varmap.items():
            if is_bool_sort(sort_id):
                self.vars[name] = z3.Bool(name)
                continue
            modulus = field_modulus_from_sort(sort_id)
            if modulus is None:
                raise ValueError("unsupported sort %s" % sort_id)
            iv = z3.Int(name)
            self.vars[name] = iv
            self.solver.add(z3.And(iv >= 0, iv < modulus))

    def _reduce(self, term: z3.ArithRef, modulus: int, boundary: bool) -> z3.ArithRef:
        """Apply modular reduction and update solver statistics."""
        if self._reducer is None:
            raise RuntimeError("mod reducer is not configured")
        self._stats["reductions_total"] += 1
        if boundary:
            self._stats["reduction_boundaries"] += 1
        else:
            self._stats["reduction_nonboundary"] += 1
        return self._reducer.reduce(term, modulus)

    def _field_modulus(self, expr: FieldExpr) -> int:
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is None:
            raise ValueError("expected a finite-field expression")
        return modulus

    def _field_modulus_from_operands(self, *exprs: FieldExpr) -> int:
        for expr in exprs:
            modulus = infer_field_modulus(expr, self.var_sorts)
            if modulus is not None:
                return modulus
        raise ValueError("expected finite-field operands")

    def _maybe_reduce(
        self,
        expr: FieldExpr,
        term: z3.ArithRef,
        modulus: int,
        schedule: str,
        at_boundary: bool = False,
        before_kernel: bool = False,
    ) -> z3.ArithRef:
        """Apply reduction iff the current schedule and metadata request it."""
        scheduler = ReductionScheduler(schedule=schedule)
        stats = None
        if self._metadata is not None:
            stats = self._metadata.stats_by_key.get(expr_key(expr))
        if scheduler.should_reduce(
            expr,
            modulus,
            stats,
            at_boundary=at_boundary,
            before_kernel=before_kernel,
        ):
            return self._reduce(term, modulus, boundary=at_boundary)
        return term

    def _pow(
        self,
        expr: FieldPow,
        base: z3.ArithRef,
        modulus: int,
        schedule: str,
    ) -> z3.ArithRef:
        """Exponentiation by squaring with schedule-aware intermediate reduction."""
        result = z3.IntVal(1)
        running_base = self._maybe_reduce(
            expr.base, base, modulus, schedule=schedule, before_kernel=True
        )
        exponent = expr.exponent
        while exponent > 0:
            if exponent & 1:
                result = self._maybe_reduce(
                    expr,
                    result * running_base,
                    modulus,
                    schedule=schedule,
                )
            exponent >>= 1
            if exponent:
                running_base = self._maybe_reduce(
                    expr,
                    running_base * running_base,
                    modulus,
                    schedule=schedule,
                )
        return result

    def _tr(
        self,
        expr: FieldExpr,
        schedule: str,
    ) -> z3.ExprRef:  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(0)
            for arg in expr.args:
                total = total + self._tr(arg, schedule=schedule)
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(1)
            for arg in expr.args:
                total = total * self._tr(arg, schedule=schedule)
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldEq):
            left = self._tr(expr.left, schedule=schedule)
            right = self._tr(expr.right, schedule=schedule)
            left_sort = infer_expr_sort(expr.left, self.var_sorts)
            right_sort = infer_expr_sort(expr.right, self.var_sorts)
            if left_sort == "bool" or right_sort == "bool":
                # Boolean equality appears after preprocessing in implication gadgets.
                return left == right
            modulus = self._field_modulus_from_operands(expr.left, expr.right)
            return self._maybe_reduce(
                expr.left,
                left,
                modulus,
                schedule=schedule,
                at_boundary=True,
                before_kernel=True,
            ) == self._maybe_reduce(
                expr.right,
                right,
                modulus,
                schedule=schedule,
                at_boundary=True,
                before_kernel=True,
            )

        if isinstance(expr, FieldVar):
            return self.vars[expr.name]

        if isinstance(expr, FieldConst):
            if expr.modulus is None:
                raise ValueError("field constants must carry a modulus")
            if not 0 <= expr.value < expr.modulus:
                raise ValueError("constant outside field range")
            return z3.IntVal(expr.value)

        if isinstance(expr, FieldSub):
            modulus = self._field_modulus(expr)
            total = self._tr(expr.args[0], schedule=schedule)
            for arg in expr.args[1:]:
                total = total - self._tr(arg, schedule=schedule)
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            total = -self._tr(expr.arg, schedule=schedule)
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            base = self._tr(expr.base, schedule=schedule)
            return self._pow(expr, base, modulus, schedule=schedule)

        if isinstance(expr, FieldDiv):
            raise ValueError(
                "Finite-field division is unsupported without an explicit nonzero side condition"
            )

        if isinstance(expr, BoolOr):
            return z3.Or(*[self._tr(arg, schedule=schedule) for arg in expr.args])

        if isinstance(expr, BoolAnd):
            return z3.And(*[self._tr(arg, schedule=schedule) for arg in expr.args])

        if isinstance(expr, BoolXor):
            args = [self._tr(arg, schedule=schedule) for arg in expr.args]
            return functools.reduce(z3.Xor, args)

        if isinstance(expr, BoolNot):
            return z3.Not(self._tr(expr.arg, schedule=schedule))

        if isinstance(expr, BoolImplies):
            return z3.Implies(
                self._tr(expr.antecedent, schedule=schedule),
                self._tr(expr.consequent, schedule=schedule),
            )

        if isinstance(expr, BoolIte):
            cond = self._tr(expr.cond, schedule=schedule)
            then_expr = self._tr(expr.then_expr, schedule=schedule)
            else_expr = self._tr(expr.else_expr, schedule=schedule)
            sort_id = infer_expr_sort(expr, self.var_sorts)
            ite_term = z3.If(cond, then_expr, else_expr)
            if sort_id is not None and not is_bool_sort(sort_id):
                modulus = self._field_modulus(expr)
                return self._maybe_reduce(
                    expr,
                    ite_term,
                    modulus,
                    schedule=schedule,
                    at_boundary=True,
                )
            return ite_term

        if isinstance(expr, BoolVar):
            return self.vars[expr.name]

        if isinstance(expr, BoolConst):
            return z3.BoolVal(expr.value)

        raise TypeError("unknown AST node %s" % type(expr).__name__)
