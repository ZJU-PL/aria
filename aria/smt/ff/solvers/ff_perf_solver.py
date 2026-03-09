#!/usr/bin/env python3
"""Performance-oriented finite-field solver backend.

This backend combines adaptive modulo-reduction scheduling (ARS) with
prime-structured modular kernels (PSK). The refinement loop is intentionally
hybrid:

1. translate the formula to integer arithmetic with selected modulo cuts;
2. validate candidate SAT models under exact GF(p) semantics;
3. localize the failure to one polynomial partition when possible;
4. learn exact partition lemmas before defining more cut semantics;
5. fall back to partition-scoped cut materialization only when lemma-based
   refinement does not make progress.

The current implementation is designed as a production-oriented research
prototype. It exposes solver stats and a per-round trace so experiment scripts
can observe how much progress comes from partition lemmas, cut definitions, and
 bounded nonlinear local solving.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import z3

from ..core.ff_ast import (
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
from ..core.ff_algebra import FFLocalAlgebraicReasoner
from ..core.ff_ir import FFIRMetadata, build_ir_metadata, expr_key
from ..core.ff_modkernels import ModKernelSelector, ModReducer
from ..core.ff_numbertheory import is_probable_prime
from ..core.ff_poly import partition_polynomial_assertions
from ..frontend.ff_preprocess import preprocess_formula
from ..core.ff_reduction_scheduler import ReductionScheduler, stricter_schedule


@dataclass
class _CutState:
    """State for one modulo-aware abstraction cut."""

    key: Tuple[object, ...]
    expr: FieldExpr
    modulus: int
    var_name: str
    defined: bool = False
    failures: int = 0


@dataclass
class _ValidationResult:
    """Validation outcome for one SAT candidate model.

    ``failed_assertions`` keeps the exact assertions falsified by the current
    abstract SAT model, while ``mismatches`` tracks undefined cuts whose model
    values disagree with exact finite-field evaluation. The solver uses the
    former to choose a failing partition and the latter as a cut-definition
    fallback inside that partition.
    """

    mismatches: List[_CutState]
    failed_assertions: List[FieldExpr]


@dataclass
class _PartitionState:
    """Refinement state for one polynomial partition.

    A partition is the connected component of polynomial assertions that share
    variables. Refinement is intentionally tracked at partition granularity so
    the solver can report where exactness was increased and avoid treating all
    mismatches as one global pool.
    """

    key: Tuple[object, ...]
    modulus: int
    assertion_indices: Tuple[int, ...]
    variables: Tuple[str, ...]
    failure_count: int = 0
    lemma_rounds: int = 0
    cut_rounds: int = 0
    exactness_level: int = 0


class FFPerfSolver:
    """Finite-field solver using ARS + PSK with local CEGAR refinement.

    The solver translates QF_FF formulas to integer arithmetic while:
        - delaying modulo operations with a schedule-aware policy;
        - selecting modulus-specific reduction kernels;
        - abstracting selected field subterms with bounded cut variables;
        - validating SAT models in exact GF(p);
        - choosing one failing polynomial partition for the next refinement;
        - learning exact local algebraic lemmas for that partition first;
        - escalating to partition-scoped cut definitions only when needed;
        - escalating schedules on ``unknown`` when recovery is enabled;
        - falling back to ``FFIntSolver`` as a safety net.

    The local algebra layer currently combines:
        - affine elimination on small partitions;
        - bounded nonlinear partition search on small moduli;
        - exact root/root-set/relation/contradiction lemmas.

    This is not yet a complete symbolic nonlinear subsolver, but it gives the
    CEGAR loop a more meaningful refinement step than simply defining the next
    mismatching cut.
    """

    def __init__(
        self,
        schedule: str = "balanced",
        kernel_mode: str = "auto",
        recovery: bool = True,
        cegar: bool = True,
        max_refinement_rounds: int = 6,
        cut_seed_budget: int = 4,
        cut_refine_budget: int = 2,
        lemma_refine_budget: int = 2,
        max_nonlinear_partition_eqs: int = 4,
        max_nonlinear_partition_vars: int = 4,
        max_nonlinear_modulus: int = 257,
        max_nonlinear_search_space: int = 4096,
        max_nonlinear_work_budget: int = 8192,
        rootset_budget: int = 4,
    ):
        env_schedule = os.getenv("ARIA_FF_SCHEDULE")
        env_kernel = os.getenv("ARIA_FF_KERNEL_MODE")
        env_max_eqs = os.getenv("ARIA_FF_MAX_NONLINEAR_EQS")
        env_max_vars = os.getenv("ARIA_FF_MAX_NONLINEAR_VARS")
        env_max_modulus = os.getenv("ARIA_FF_MAX_NONLINEAR_MODULUS")
        env_search_space = os.getenv("ARIA_FF_MAX_NONLINEAR_SEARCH_SPACE")
        env_work_budget = os.getenv("ARIA_FF_MAX_NONLINEAR_WORK_BUDGET")
        env_rootset_budget = os.getenv("ARIA_FF_ROOTSET_BUDGET")
        if env_schedule:
            schedule = env_schedule.strip()
        if env_kernel:
            kernel_mode = env_kernel.strip()
        if env_max_eqs:
            max_nonlinear_partition_eqs = int(env_max_eqs.strip())
        if env_max_vars:
            max_nonlinear_partition_vars = int(env_max_vars.strip())
        if env_max_modulus:
            max_nonlinear_modulus = int(env_max_modulus.strip())
        if env_search_space:
            max_nonlinear_search_space = int(env_search_space.strip())
        if env_work_budget:
            max_nonlinear_work_budget = int(env_work_budget.strip())
        if env_rootset_budget:
            rootset_budget = int(env_rootset_budget.strip())

        self.initial_schedule = schedule
        self.kernel_mode = kernel_mode
        self.recovery = recovery
        self.cegar = cegar
        self.max_refinement_rounds = max(0, max_refinement_rounds)
        self.cut_seed_budget = max(0, cut_seed_budget)
        self.cut_refine_budget = max(1, cut_refine_budget)
        self.lemma_refine_budget = max(1, lemma_refine_budget)
        self.max_nonlinear_partition_eqs = max(1, max_nonlinear_partition_eqs)
        self.max_nonlinear_partition_vars = max(1, max_nonlinear_partition_vars)
        self.max_nonlinear_modulus = max(2, max_nonlinear_modulus)
        self.max_nonlinear_search_space = max(2, max_nonlinear_search_space)
        self.max_nonlinear_work_budget = max(16, max_nonlinear_work_budget)
        self.rootset_budget = max(2, rootset_budget)
        self.solver = z3.SolverFor("QF_NIA")
        self.vars: Dict[str, z3.ExprRef] = {}
        self.var_sorts: Dict[str, str] = {}
        self._metadata: Optional[FFIRMetadata] = None
        self._reducer: Optional[ModReducer] = None
        self._stats: Dict[str, int] = {}
        self._cut_pool: Dict[Tuple[object, ...], FieldExpr] = {}
        self._cuts: Dict[Tuple[object, ...], _CutState] = {}
        self._cut_z3_vars: Dict[Tuple[object, ...], z3.ArithRef] = {}
        self._next_cut_id = 0
        self._reasoner = FFLocalAlgebraicReasoner(
            max_nonlinear_partition_eqs=self.max_nonlinear_partition_eqs,
            max_nonlinear_partition_vars=self.max_nonlinear_partition_vars,
            max_nonlinear_modulus=self.max_nonlinear_modulus,
            max_nonlinear_search_space=self.max_nonlinear_search_space,
            max_nonlinear_work_budget=self.max_nonlinear_work_budget,
            rootset_budget=self.rootset_budget,
        )
        self._learned_lemmas: List[FieldExpr] = []
        self._learned_lemma_keys: Set[Tuple[object, ...]] = set()
        self._partition_order: Dict[Tuple[object, ...], Tuple[int, int]] = {}
        self._partitions: Dict[Tuple[object, ...], _PartitionState] = {}
        self._assertion_to_partition: Dict[Tuple[object, ...], Tuple[object, ...]] = {}
        self._expr_to_partitions: Dict[Tuple[object, ...], Set[Tuple[object, ...]]] = {}
        self._last_refined_partition: Optional[Tuple[object, ...]] = None
        self._current_assertions: Sequence[FieldExpr] = ()
        self._trace: List[Dict[str, object]] = []
        self._pending_lemma_impact = 0

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Check satisfiability with adaptive schedule fallback and local CEGAR."""
        normalized = preprocess_formula(formula)
        self._reset_stats()
        self._setup_fields(normalized.field_sizes)
        self.var_sorts = dict(normalized.variables)
        self._metadata = build_ir_metadata(normalized.assertions)
        self._current_assertions = tuple(normalized.assertions)
        self._cut_pool = self._collect_cut_pool(normalized.assertions)
        self._cuts = {}
        self._cut_z3_vars = {}
        self._next_cut_id = 0
        self._learned_lemmas = []
        self._learned_lemma_keys = set()
        self._reasoner.reset_stats()
        self._partition_order = self._build_partition_order(normalized)
        self._partitions = self._build_partitions(normalized)
        self._assertion_to_partition = self._build_assertion_partition_map(normalized)
        self._expr_to_partitions = self._build_expr_partition_map(normalized)
        self._last_refined_partition = None
        if self.cegar:
            self._seed_initial_cuts()

        schedules = self._schedule_chain(self.initial_schedule)
        last_result = z3.unknown
        for attempt, schedule in enumerate(schedules):
            self._stats["attempt"] = attempt + 1
            self._stats["fallback_attempts"] = attempt
            self._stats["schedule_%s" % schedule] = 1

            result = self._run_schedule_attempt(normalized, schedule)
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
        """Return counters from the latest run.

        The stats dictionary mixes low-level encoding counters
        (reductions/kernels/cuts) with refinement counters:
            - lemma_*
            - partition_*
            - selected_partition_*
            - partition_solver_*

        These are intended to support experiment scripts and ablations.
        """
        return dict(self._stats)

    def trace(self) -> List[Dict[str, object]]:
        """Return per-round refinement trace from the latest run.

        Each entry summarizes one refinement round or terminal solver event. The
        trace is deliberately lightweight and JSON-friendly so benchmark scripts
        can persist it without postprocessing.
        """
        return [dict(entry) for entry in self._trace]

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
        self._cut_z3_vars = {}

    def _reset_stats(self) -> None:
        self._stats = {
            "reductions_total": 0,
            "reduction_boundaries": 0,
            "reduction_nonboundary": 0,
            "kernel_generic": 0,
            "kernel_pseudo_mersenne": 0,
            "kernel_near_power2_sparse": 0,
            "kernel_small_prime_unrolled": 0,
            "cegar_rounds": 0,
            "validation_failures": 0,
            "cuts_seeded": 0,
            "cuts_activated_dynamic": 0,
            "cuts_defined": 0,
            "cuts_active": 0,
            "cuts_defined_active": 0,
            "cut_mismatch_hits": 0,
            "validated_sat_models": 0,
            "lemmas_learned": 0,
            "lemma_zero_product": 0,
            "lemma_linear_root": 0,
            "lemma_power_zero": 0,
            "lemma_monomial_zero": 0,
            "lemma_constant_contradiction": 0,
            "lemma_affine_root": 0,
            "lemma_affine_relation": 0,
            "lemma_affine_contradiction": 0,
            "lemma_partition_root": 0,
            "lemma_partition_rootset": 0,
            "lemma_partition_relation": 0,
            "lemma_partition_contradiction": 0,
            "partitions_total": 0,
            "partitions_failed": 0,
            "partition_refinements": 0,
            "partition_switches": 0,
            "partition_lemma_rounds": 0,
            "partition_cut_rounds": 0,
            "partition_exactness_gain": 0,
            "lemma_rounds_total": 0,
            "lemma_rounds_avoided_cuts": 0,
            "lemma_rounds_led_to_unsat": 0,
            "lemma_rounds_led_to_sat": 0,
            "useful_lemmas": 0,
            "cuts_avoided_by_lemmas": 0,
            "selected_partition_size": 0,
            "selected_partition_variables": 0,
            "selected_partition_modulus": 0,
            "partition_solver_hits": 0,
            "partition_cache_hits": 0,
            "partition_cache_misses": 0,
            "partition_solver_calls": 0,
            "partition_solver_enumerations": 0,
            "partition_solver_search_nodes": 0,
            "partition_solver_budget_abort": 0,
            "partition_solver_small_space_skip": 0,
        }
        self._trace = []
        self._pending_lemma_impact = 0

    def _build_partition_order(
        self, formula: ParsedFormula
    ) -> Dict[Tuple[object, ...], Tuple[int, int]]:
        order: Dict[Tuple[object, ...], Tuple[int, int]] = {}
        for partition in partition_polynomial_assertions(
            formula.assertions, formula.variables
        ):
            score = (len(partition.assertion_indices), len(partition.variables))
            for assertion_idx in partition.assertion_indices:
                assertion = formula.assertions[assertion_idx]
                order[expr_key(assertion)] = score
        return order

    def _build_partitions(
        self, formula: ParsedFormula
    ) -> Dict[Tuple[object, ...], _PartitionState]:
        partitions: Dict[Tuple[object, ...], _PartitionState] = {}
        for partition in partition_polynomial_assertions(
            formula.assertions, formula.variables
        ):
            key = (
                "partition",
                partition.modulus,
                partition.assertion_indices,
                partition.variables,
            )
            partitions[key] = _PartitionState(
                key=key,
                modulus=partition.modulus,
                assertion_indices=partition.assertion_indices,
                variables=partition.variables,
            )
        self._stats["partitions_total"] = len(partitions)
        return partitions

    def _build_assertion_partition_map(
        self, formula: ParsedFormula
    ) -> Dict[Tuple[object, ...], Tuple[object, ...]]:
        mapping: Dict[Tuple[object, ...], Tuple[object, ...]] = {}
        for partition in self._partitions.values():
            for assertion_idx in partition.assertion_indices:
                mapping[expr_key(formula.assertions[assertion_idx])] = partition.key
        return mapping

    def _build_expr_partition_map(
        self, formula: ParsedFormula
    ) -> Dict[Tuple[object, ...], Set[Tuple[object, ...]]]:
        mapping: Dict[Tuple[object, ...], Set[Tuple[object, ...]]] = {}
        for partition in self._partitions.values():
            for assertion_idx in partition.assertion_indices:
                for node in self._walk_expr(formula.assertions[assertion_idx]):
                    mapping.setdefault(expr_key(node), set()).add(partition.key)
        return mapping

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

    def _declare_cut_vars(self) -> None:
        """Declare bounded integer proxies for active cut expressions."""
        for cut in self._cuts.values():
            iv = z3.Int(cut.var_name)
            self._cut_z3_vars[cut.key] = iv
            self.solver.add(z3.And(iv >= 0, iv < cut.modulus))

    def _add_defined_cut_constraints(self, schedule: str) -> None:
        """Materialize the exact modulo semantics for defined cuts only."""
        for cut in self._cuts.values():
            if not cut.defined:
                continue
            rhs = self._tr(cut.expr, schedule=schedule, defining_key=cut.key)
            rhs = self._maybe_reduce(
                cut.expr,
                rhs,
                cut.modulus,
                schedule=schedule,
                at_boundary=True,
                before_kernel=True,
            )
            self.solver.add(self._cut_z3_vars[cut.key] == rhs)

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

    def _run_schedule_attempt(
        self, formula: ParsedFormula, schedule: str
    ) -> z3.CheckSatResult:
        """Solve one schedule attempt with partition-driven CEGAR.

        One refinement round has the following shape:
            1. solve the current abstraction;
            2. if SAT, validate exactly in GF(p);
            3. select one failing polynomial partition;
            4. try exact partition lemmas;
            5. if no lemma is learned, increase exactness with partition-scoped
               cut definitions or new cuts.
        """
        if not self.cegar:
            self._reset_solver()
            self._declare_vars(formula.variables)
            self._setup_kernels(formula.field_sizes)
            for assertion in formula.assertions:
                self.solver.add(self._tr(assertion, schedule=schedule))
            return self.solver.check()

        for round_idx in range(self.max_refinement_rounds + 1):
            self._stats["cegar_rounds"] = max(
                self._stats.get("cegar_rounds", 0), round_idx + 1
            )
            self._reset_solver()
            self._declare_vars(formula.variables)
            self._setup_kernels(formula.field_sizes)
            self._declare_cut_vars()
            self._add_defined_cut_constraints(schedule)
            for lemma in self._learned_lemmas:
                self.solver.add(self._tr(lemma, schedule=schedule))
            for assertion in formula.assertions:
                self.solver.add(self._tr(assertion, schedule=schedule))

            self._stats["cuts_active"] = len(self._cuts)
            self._stats["cuts_defined_active"] = sum(
                1 for cut in self._cuts.values() if cut.defined
            )

            result = self.solver.check()
            if result != z3.sat:
                self._record_terminal_result(result, schedule, round_idx)
                return result

            validation = self._validate_current_model(formula.assertions)
            if not validation.failed_assertions:
                self._stats["validated_sat_models"] += 1
                self._record_terminal_result(z3.sat, schedule, round_idx)
                return z3.sat

            self._stats["validation_failures"] += 1
            refinement = self._select_refinement_partition(validation)
            learned_now = self._learn_explanation_lemmas(refinement["assertions"])
            if learned_now:
                self._mark_partition_lemma_progress(refinement["partition"])
                self._stats["lemma_rounds_total"] += 1
                self._stats["lemma_rounds_avoided_cuts"] += 1
                self._stats["cuts_avoided_by_lemmas"] += learned_now
                self._pending_lemma_impact = learned_now
                self._append_round_trace(
                    schedule,
                    round_idx,
                    refinement,
                    validation,
                    learned_now=learned_now,
                    cuts_defined=0,
                    cuts_activated=0,
                )
                continue
            if round_idx >= self.max_refinement_rounds:
                break
            cut_progress = self._refine_cuts(
                refinement["mismatches"], refinement["partition"]
            )
            self._append_round_trace(
                schedule,
                round_idx,
                refinement,
                validation,
                learned_now=0,
                cuts_defined=cut_progress[1],
                cuts_activated=cut_progress[2],
            )
            if not cut_progress[0]:
                break
            self._mark_partition_cut_progress(refinement["partition"])
            self._pending_lemma_impact = 0

        return z3.unknown

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
        defining_key: Optional[Tuple[object, ...]] = None,
    ) -> z3.ExprRef:  # pylint: disable=too-many-return-statements,too-many-branches
        sort_id = infer_expr_sort(expr, self.var_sorts)
        key = expr_key(expr)
        cut = self._cuts.get(key)
        if (
            cut is not None
            and key != defining_key
            and sort_id is not None
            and not is_bool_sort(sort_id)
        ):
            return self._cut_z3_vars[cut.key]

        if isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(0)
            for arg in expr.args:
                total = total + self._tr(
                    arg, schedule=schedule, defining_key=defining_key
                )
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(1)
            for arg in expr.args:
                total = total * self._tr(
                    arg, schedule=schedule, defining_key=defining_key
                )
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldEq):
            left = self._tr(expr.left, schedule=schedule, defining_key=defining_key)
            right = self._tr(expr.right, schedule=schedule, defining_key=defining_key)
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
            total = self._tr(
                expr.args[0], schedule=schedule, defining_key=defining_key
            )
            for arg in expr.args[1:]:
                total = total - self._tr(
                    arg, schedule=schedule, defining_key=defining_key
                )
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            total = -self._tr(expr.arg, schedule=schedule, defining_key=defining_key)
            return self._maybe_reduce(expr, total, modulus, schedule=schedule)

        if isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            base = self._tr(expr.base, schedule=schedule, defining_key=defining_key)
            return self._pow(expr, base, modulus, schedule=schedule)

        if isinstance(expr, FieldDiv):
            raise ValueError(
                "Finite-field division is unsupported without an explicit nonzero side condition"
            )

        if isinstance(expr, BoolOr):
            return z3.Or(
                *[
                    self._tr(arg, schedule=schedule, defining_key=defining_key)
                    for arg in expr.args
                ]
            )

        if isinstance(expr, BoolAnd):
            return z3.And(
                *[
                    self._tr(arg, schedule=schedule, defining_key=defining_key)
                    for arg in expr.args
                ]
            )

        if isinstance(expr, BoolXor):
            args = [
                self._tr(arg, schedule=schedule, defining_key=defining_key)
                for arg in expr.args
            ]
            return functools.reduce(z3.Xor, args)

        if isinstance(expr, BoolNot):
            return z3.Not(
                self._tr(expr.arg, schedule=schedule, defining_key=defining_key)
            )

        if isinstance(expr, BoolImplies):
            return z3.Implies(
                self._tr(
                    expr.antecedent, schedule=schedule, defining_key=defining_key
                ),
                self._tr(
                    expr.consequent, schedule=schedule, defining_key=defining_key
                ),
            )

        if isinstance(expr, BoolIte):
            cond = self._tr(expr.cond, schedule=schedule, defining_key=defining_key)
            then_expr = self._tr(
                expr.then_expr, schedule=schedule, defining_key=defining_key
            )
            else_expr = self._tr(
                expr.else_expr, schedule=schedule, defining_key=defining_key
            )
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

    def _collect_cut_pool(
        self, assertions: Sequence[FieldExpr]
    ) -> Dict[Tuple[object, ...], FieldExpr]:
        """Index candidate field subterms that can be abstracted by cuts."""
        pool: Dict[Tuple[object, ...], FieldExpr] = {}
        for assertion in assertions:
            for expr in self._walk_expr(assertion):
                if isinstance(expr, (FieldConst, FieldVar, FieldEq)):
                    continue
                sort_id = infer_expr_sort(expr, self.var_sorts)
                if sort_id is None or is_bool_sort(sort_id):
                    continue
                modulus = infer_field_modulus(expr, self.var_sorts)
                if modulus is None:
                    continue
                key = expr_key(expr)
                pool.setdefault(key, expr)
        return pool

    def _seed_initial_cuts(self) -> None:
        """Activate a small initial abstraction budget on hard field subterms."""
        scored: List[Tuple[int, FieldExpr]] = []
        for key, expr in self._cut_pool.items():
            if self._metadata is None:
                continue
            stats = self._metadata.stats_by_key.get(key)
            if stats is None:
                continue
            modulus = infer_field_modulus(expr, self.var_sorts)
            if modulus is None:
                continue
            field_bits = max(1, (modulus - 1).bit_length())
            eligible = (
                stats.nonlinear
                or stats.fanout >= 2
                or stats.est_bits >= field_bits * 2
            )
            if not eligible:
                continue
            scored.append((self._cut_priority(expr, dynamic_weight=0), expr))

        scored.sort(key=lambda item: item[0], reverse=True)
        for _score, expr in scored[: self.cut_seed_budget]:
            if self._activate_cut(expr):
                self._stats["cuts_seeded"] += 1

        self._stats["cuts_active"] = len(self._cuts)

    def _cut_priority(self, expr: FieldExpr, dynamic_weight: int) -> int:
        """Return a structural priority score for cut activation/refinement."""
        key = expr_key(expr)
        score = dynamic_weight * 50
        if self._metadata is None:
            return score
        stats = self._metadata.stats_by_key.get(key)
        if stats is None:
            return score
        score += stats.depth
        score += stats.fanout * 8
        score += 12 if stats.nonlinear else 0
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is not None:
            field_bits = max(1, (modulus - 1).bit_length())
            score += stats.est_bits // field_bits
        return score

    def _activate_cut(self, expr: FieldExpr) -> bool:
        """Register a new abstract cut variable for one field expression."""
        key = expr_key(expr)
        if key in self._cuts:
            return False
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is None:
            return False
        cut = _CutState(
            key=key,
            expr=expr,
            modulus=modulus,
            var_name="ff_cut_%d" % self._next_cut_id,
        )
        self._next_cut_id += 1
        self._cuts[key] = cut
        self._stats["cuts_active"] = len(self._cuts)
        return True

    def _validate_current_model(
        self, assertions: Sequence[FieldExpr]
    ) -> _ValidationResult:
        """Validate the current SAT model in exact GF(p) semantics."""
        model = self.solver.model()
        exact_cache: Dict[Tuple[object, ...], object] = {}
        mismatches: Dict[Tuple[object, ...], _CutState] = {}
        failed_assertions: List[FieldExpr] = []

        for assertion in assertions:
            if self._eval_exact(assertion, model, exact_cache):
                continue
            failed_assertions.append(assertion)
            for cut in self._collect_mismatching_cuts(assertion, model, exact_cache):
                mismatches[cut.key] = cut

        self._stats["cut_mismatch_hits"] += len(mismatches)
        return _ValidationResult(
            mismatches=sorted(
                mismatches.values(),
                key=lambda cut: self._cut_priority(
                    cut.expr, dynamic_weight=cut.failures + 1
                ),
                reverse=True,
            ),
            failed_assertions=failed_assertions,
        )

    def _learn_explanation_lemmas(
        self, failed_assertions: Sequence[FieldExpr]
    ) -> int:
        """Learn exact algebraic lemmas before defining more cuts."""
        learned_now = 0
        for lemma in self._reasoner.derive_partition_lemmas(
            failed_assertions, self.var_sorts
        ):
                if self._record_learned_lemma(lemma):
                    learned_now += 1
                    if learned_now >= self.lemma_refine_budget:
                        self._sync_reasoner_stats()
                        return learned_now

        ordered = sorted(
            failed_assertions,
            key=lambda assertion: self._partition_order.get(
                expr_key(assertion), (len(failed_assertions), 0)
            ),
        )
        for assertion in ordered:
            for lemma in self._reasoner.derive_lemmas(assertion, self.var_sorts):
                if self._record_learned_lemma(lemma):
                    learned_now += 1
                    if learned_now >= self.lemma_refine_budget:
                        self._sync_reasoner_stats()
                        return learned_now
        self._sync_reasoner_stats()
        return learned_now

    def _record_learned_lemma(self, lemma) -> bool:
        key = expr_key(lemma.expr)
        if key in self._learned_lemma_keys:
            return False
        self._learned_lemma_keys.add(key)
        self._learned_lemmas.append(lemma.expr)
        self._stats["lemmas_learned"] += 1
        stat_key = "lemma_%s" % lemma.kind.replace("-", "_")
        self._stats[stat_key] = self._stats.get(stat_key, 0) + 1
        return True

    def _select_refinement_partition(
        self, validation: _ValidationResult
    ) -> Dict[str, object]:
        """Choose one failing partition to refine, or fall back globally.

        The score is intentionally simple and stable: prefer partitions with
        more failing assertions and mismatching cuts, then use previous failure
        count and partition size as tie-breakers.
        """
        partition_failures: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for assertion in validation.failed_assertions:
            part_key = self._assertion_to_partition.get(expr_key(assertion))
            if part_key is None:
                continue
            bucket = partition_failures.setdefault(
                part_key, {"assertions": [], "mismatches": []}
            )
            bucket["assertions"].append(assertion)

        for mismatch in validation.mismatches:
            for part_key in self._expr_to_partitions.get(mismatch.key, set()):
                bucket = partition_failures.setdefault(
                    part_key, {"assertions": [], "mismatches": []}
                )
                bucket["mismatches"].append(mismatch)

        self._stats["partitions_failed"] = len(partition_failures)
        if not partition_failures:
            return {
                "partition": None,
                "assertions": list(validation.failed_assertions),
                "mismatches": list(validation.mismatches),
            }

        best_key = max(
            partition_failures,
            key=lambda key: self._partition_refinement_score(
                self._partitions[key],
                len(partition_failures[key]["assertions"]),
                len(partition_failures[key]["mismatches"]),
            ),
        )
        if self._last_refined_partition != best_key:
            if self._last_refined_partition is not None:
                self._stats["partition_switches"] += 1
            self._last_refined_partition = best_key

        bucket = partition_failures[best_key]
        partition = self._partitions[best_key]
        partition.failure_count += 1
        self._stats["partition_refinements"] += 1
        self._stats["selected_partition_size"] = len(partition.assertion_indices)
        self._stats["selected_partition_variables"] = len(partition.variables)
        self._stats["selected_partition_modulus"] = partition.modulus
        return {
            "partition": partition,
            "assertions": bucket["assertions"],
            "mismatches": bucket["mismatches"],
        }

    def _partition_refinement_score(
        self,
        partition: _PartitionState,
        failed_assertions: int,
        mismatch_count: int,
    ) -> Tuple[int, int, int, int, int]:
        """Score one failing partition for the next refinement round."""
        return (
            failed_assertions + mismatch_count,
            partition.failure_count,
            len(partition.assertion_indices),
            len(partition.variables),
            -partition.exactness_level,
        )

    def _mark_partition_lemma_progress(
        self, partition: Optional[_PartitionState]
    ) -> None:
        if partition is None:
            return
        partition.lemma_rounds += 1
        partition.exactness_level += 1
        self._stats["partition_lemma_rounds"] += 1
        self._stats["partition_exactness_gain"] += 1

    def _mark_partition_cut_progress(
        self, partition: Optional[_PartitionState]
    ) -> None:
        if partition is None:
            return
        partition.cut_rounds += 1
        partition.exactness_level += 1
        self._stats["partition_cut_rounds"] += 1
        self._stats["partition_exactness_gain"] += 1

    def _record_terminal_result(
        self, result: z3.CheckSatResult, schedule: str, round_idx: int
    ) -> None:
        """Attribute terminal outcomes to the most recent lemma round if any."""
        if self._pending_lemma_impact > 0:
            if result == z3.unsat:
                self._stats["lemma_rounds_led_to_unsat"] += 1
                self._stats["useful_lemmas"] += self._pending_lemma_impact
            elif result == z3.sat:
                self._stats["lemma_rounds_led_to_sat"] += 1
                self._stats["useful_lemmas"] += self._pending_lemma_impact
        self._append_round_trace(
            schedule,
            round_idx,
            {"partition": None, "assertions": [], "mismatches": []},
            None,
            learned_now=0,
            cuts_defined=0,
            cuts_activated=0,
            terminal_result=str(result),
        )
        self._pending_lemma_impact = 0

    def _append_round_trace(
        self,
        schedule: str,
        round_idx: int,
        refinement: Dict[str, object],
        validation: Optional[_ValidationResult],
        learned_now: int,
        cuts_defined: int,
        cuts_activated: int,
        terminal_result: Optional[str] = None,
    ) -> None:
        """Append a JSON-friendly trace entry for one refinement round."""
        partition = refinement.get("partition")
        entry: Dict[str, object] = {
            "schedule": schedule,
            "round": round_idx + 1,
            "failed_assertions": 0 if validation is None else len(validation.failed_assertions),
            "failed_partitions": self._stats.get("partitions_failed", 0),
            "learned_lemmas": learned_now,
            "cuts_defined": cuts_defined,
            "cuts_activated": cuts_activated,
            "terminal_result": terminal_result,
        }
        if partition is not None:
            entry["partition_key"] = partition.key
            entry["partition_size"] = len(partition.assertion_indices)
            entry["partition_variables"] = len(partition.variables)
            entry["partition_modulus"] = partition.modulus
            entry["partition_exactness_level"] = partition.exactness_level
        self._trace.append(entry)

    def _sync_reasoner_stats(self) -> None:
        """Merge local algebra stats into the solver-visible stats dictionary."""
        for key, value in self._reasoner.stats().items():
            self._stats[key] = value
        self._stats["partition_solver_hits"] = (
            self._stats.get("partition_solver_enumerations", 0)
            + self._stats.get("partition_cache_hits", 0)
        )

    def _collect_mismatching_cuts(
        self,
        expr: FieldExpr,
        model: z3.ModelRef,
        exact_cache: Dict[Tuple[object, ...], object],
    ) -> List[_CutState]:
        """Return undefined cuts whose abstract model value disagrees with GF(p)."""
        mismatches: Dict[Tuple[object, ...], _CutState] = {}
        for node in self._walk_expr(expr):
            key = expr_key(node)
            cut = self._cuts.get(key)
            if cut is None or cut.defined:
                continue
            model_val = model.eval(
                self._cut_z3_vars[cut.key], model_completion=True
            ).as_long()
            exact_val = self._eval_exact(node, model, exact_cache)
            if isinstance(exact_val, bool):
                continue
            if model_val != exact_val:
                mismatches[key] = cut
        return list(mismatches.values())

    def _refine_cuts(
        self,
        mismatches: Sequence[_CutState],
        partition: Optional[_PartitionState] = None,
    ) -> Tuple[bool, int, int]:
        """Define or activate cuts, preferring the selected partition."""
        progress = False
        defined_now = 0
        activated_now = 0
        for cut in mismatches:
            if cut.defined:
                continue
            cut.defined = True
            cut.failures += 1
            defined_now += 1
            self._stats["cuts_defined"] += 1
            progress = True
            if defined_now >= self.cut_refine_budget:
                break

        if progress:
            self._stats["cuts_defined_active"] = sum(
                1 for cut in self._cuts.values() if cut.defined
            )
            return (True, defined_now, activated_now)

        candidate_keys: Optional[Set[Tuple[object, ...]]] = None
        if partition is not None:
            candidate_keys = set()
            for assertion_idx in partition.assertion_indices:
                for expr in self._walk_expr(
                    self._current_assertions[assertion_idx]
                ):
                    candidate_keys.add(expr_key(expr))

        scored: List[Tuple[int, FieldExpr]] = []
        for key, expr in self._cut_pool.items():
            if key in self._cuts:
                continue
            if candidate_keys is not None and key not in candidate_keys:
                continue
            scored.append((self._cut_priority(expr, dynamic_weight=0), expr))
        if not scored and candidate_keys is not None:
            for key, expr in self._cut_pool.items():
                if key in self._cuts:
                    continue
                scored.append((self._cut_priority(expr, dynamic_weight=0), expr))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _score, expr in scored[: self.cut_refine_budget]:
            if self._activate_cut(expr):
                self._stats["cuts_activated_dynamic"] += 1
                progress = True
                activated_now += 1
        return (progress, defined_now, activated_now)

    def _eval_exact(
        self,
        expr: FieldExpr,
        model: z3.ModelRef,
        cache: Dict[Tuple[object, ...], object],
    ) -> object:
        """Evaluate one expression under exact finite-field semantics."""
        key = expr_key(expr)
        if key in cache:
            return cache[key]

        if isinstance(expr, FieldVar):
            value = model.eval(self.vars[expr.name], model_completion=True).as_long()
        elif isinstance(expr, FieldConst):
            if expr.modulus is None:
                raise ValueError("field constants must carry a modulus")
            value = expr.value % expr.modulus
        elif isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            value = sum(
                int(self._eval_exact(arg, model, cache)) for arg in expr.args
            ) % modulus
        elif isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            value = 1
            for arg in expr.args:
                value = (value * int(self._eval_exact(arg, model, cache))) % modulus
        elif isinstance(expr, FieldSub):
            modulus = self._field_modulus(expr)
            value = int(self._eval_exact(expr.args[0], model, cache))
            for arg in expr.args[1:]:
                value = (value - int(self._eval_exact(arg, model, cache))) % modulus
        elif isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            value = (-int(self._eval_exact(expr.arg, model, cache))) % modulus
        elif isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            base = int(self._eval_exact(expr.base, model, cache))
            value = pow(base, expr.exponent, modulus)
        elif isinstance(expr, FieldDiv):
            raise ValueError(
                "Finite-field division is unsupported without an explicit nonzero side condition"
            )
        elif isinstance(expr, FieldEq):
            left = self._eval_exact(expr.left, model, cache)
            right = self._eval_exact(expr.right, model, cache)
            value = left == right
        elif isinstance(expr, BoolOr):
            value = any(bool(self._eval_exact(arg, model, cache)) for arg in expr.args)
        elif isinstance(expr, BoolAnd):
            value = all(bool(self._eval_exact(arg, model, cache)) for arg in expr.args)
        elif isinstance(expr, BoolXor):
            parity = False
            for arg in expr.args:
                parity = parity ^ bool(self._eval_exact(arg, model, cache))
            value = parity
        elif isinstance(expr, BoolNot):
            value = not bool(self._eval_exact(expr.arg, model, cache))
        elif isinstance(expr, BoolImplies):
            value = (not bool(self._eval_exact(expr.antecedent, model, cache))) or bool(
                self._eval_exact(expr.consequent, model, cache)
            )
        elif isinstance(expr, BoolIte):
            cond = bool(self._eval_exact(expr.cond, model, cache))
            chosen = expr.then_expr if cond else expr.else_expr
            value = self._eval_exact(chosen, model, cache)
        elif isinstance(expr, BoolVar):
            value = z3.is_true(model.eval(self.vars[expr.name], model_completion=True))
        elif isinstance(expr, BoolConst):
            value = expr.value
        else:
            raise TypeError("unknown AST node %s" % type(expr).__name__)

        cache[key] = value
        return value

    def _walk_expr(self, expr: FieldExpr) -> Sequence[FieldExpr]:
        """Yield one node and all descendants in deterministic preorder."""
        nodes = [expr]
        if isinstance(expr, (FieldAdd, FieldMul, FieldSub, BoolAnd, BoolOr, BoolXor)):
            for arg in expr.args:
                nodes.extend(self._walk_expr(arg))
            return nodes
        if isinstance(expr, (FieldNeg, FieldPow, BoolNot)):
            child = expr.arg if isinstance(expr, (FieldNeg, BoolNot)) else expr.base
            nodes.extend(self._walk_expr(child))
            return nodes
        if isinstance(expr, FieldEq):
            nodes.extend(self._walk_expr(expr.left))
            nodes.extend(self._walk_expr(expr.right))
            return nodes
        if isinstance(expr, BoolImplies):
            nodes.extend(self._walk_expr(expr.antecedent))
            nodes.extend(self._walk_expr(expr.consequent))
            return nodes
        if isinstance(expr, BoolIte):
            nodes.extend(self._walk_expr(expr.cond))
            nodes.extend(self._walk_expr(expr.then_expr))
            nodes.extend(self._walk_expr(expr.else_expr))
            return nodes
        if isinstance(expr, FieldDiv):
            nodes.extend(self._walk_expr(expr.num))
            nodes.extend(self._walk_expr(expr.denom))
            return nodes
        return nodes
