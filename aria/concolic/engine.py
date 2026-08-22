"""Public native concolic trace and exploration engine."""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

import z3

from .instrumentation import instrument_callable
from .invocation import build_call_arguments
from .coverage import CoverageAccumulator, CoverageSession
from .gaps import explain_gaps
from .models import (
    ConcolicResult,
    CoverageGate,
    CoverageSnapshot,
    Diagnostic,
    DiagnosticCode,
    ExecutionOutcome,
    ExecutionTrace,
    Frontier,
)
from .package import PackageInstrumentor
from .runtime import session_scope
from .sampling import ConstraintSampler
from .structured import expand_input_shapes, materialize_candidate


class ExplorationStrategy(str, Enum):
    """Frontier traversal policies."""

    BREADTH_FIRST = "breadth-first"
    DEPTH_FIRST = "depth-first"
    COVERAGE = "coverage"


InputConstraint = Callable[[Mapping[str, z3.ExprRef]], z3.BoolRef]


@dataclass(frozen=True)
class ConcolicOptions:
    """Resource and search configuration for native concolic exploration."""

    max_iterations: int = 100
    max_frontiers: int = 10_000
    samples_per_frontier: int = 4
    random_seed: Optional[int] = None
    strategy: ExplorationStrategy = ExplorationStrategy.COVERAGE
    stop_on_error: bool = False
    input_constraint: Optional[InputConstraint] = None
    isolate: bool = False
    execution_timeout: float = 10.0
    worker_start_method: Optional[str] = None
    max_yields: int = 1_000
    memory_limit_mb: Optional[int] = None
    max_output_chars: int = 100_000
    solver_timeout: float = 5.0
    campaign_timeout: Optional[float] = None
    max_path_depth: int = 256
    max_constraint_nodes: int = 10_000
    max_inputs_per_path: int = 2
    measure_coverage: bool = False
    coverage_sources: tuple[str, ...] = ()
    coverage_include: tuple[str, ...] = ()
    coverage_omit: tuple[str, ...] = ()
    instrument_packages: tuple[str, ...] = ()
    instrument_include: tuple[str, ...] = ()
    instrument_omit: tuple[str, ...] = ()
    instrument_import_submodules: bool = True
    coverage_fail_under_lines: Optional[float] = None
    coverage_fail_under_statements: Optional[float] = None
    coverage_fail_under_functions: Optional[float] = None
    coverage_fail_under_branches: Optional[float] = None
    expand_input_shapes: bool = False
    max_shape_variants: int = 32
    explain_gaps: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_frontiers <= 0:
            raise ValueError("max_frontiers must be positive")
        if self.samples_per_frontier <= 0:
            raise ValueError("samples_per_frontier must be positive")
        if self.execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive")
        if self.max_yields <= 0:
            raise ValueError("max_yields must be positive")
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive when provided")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")
        if self.solver_timeout <= 0:
            raise ValueError("solver_timeout must be positive")
        if self.campaign_timeout is not None and self.campaign_timeout <= 0:
            raise ValueError("campaign_timeout must be positive when provided")
        if self.max_path_depth <= 0:
            raise ValueError("max_path_depth must be positive")
        if self.max_constraint_nodes <= 0:
            raise ValueError("max_constraint_nodes must be positive")
        if self.max_inputs_per_path <= 0:
            raise ValueError("max_inputs_per_path must be positive")
        if self.max_shape_variants <= 0:
            raise ValueError("max_shape_variants must be positive")
        for name, threshold in self.coverage_thresholds.items():
            if not 0.0 <= threshold <= 100.0:
                raise ValueError(f"{name} coverage threshold must be between 0 and 100")

    @property
    def coverage_thresholds(self) -> Dict[str, float]:
        values = {
            "lines": self.coverage_fail_under_lines,
            "statements": self.coverage_fail_under_statements,
            "functions": self.coverage_fail_under_functions,
            "branches": self.coverage_fail_under_branches,
        }
        return {name: value for name, value in values.items() if value is not None}


class NativeConcolicEngine:
    """Instrument and explore a source-available Python callable."""

    def __init__(
        self,
        target: Callable[..., Any],
        options: Optional[ConcolicOptions] = None,
        sampler: Optional[ConstraintSampler] = None,
    ) -> None:
        self.target = target
        self.options = options or ConcolicOptions()
        self.package_instrumentor = None
        if self.options.instrument_packages:
            self.package_instrumentor = PackageInstrumentor(
                self.options.instrument_packages,
                self.options.instrument_include,
                self.options.instrument_omit,
                self.options.instrument_import_submodules,
            )
            self.instrumented = self.package_instrumentor.instrument(target)
        else:
            self.instrumented = instrument_callable(target)
        self.signature = inspect.signature(target)
        self.sampler = sampler or ConstraintSampler()
        self._isolated_executor = None
        if self.options.isolate:
            from .execution import IsolatedExecutor

            if inspect.ismethod(target) and target.__self__ is not None:
                raise TypeError("isolated execution does not accept bound methods")
            self._isolated_executor = IsolatedExecutor(
                target.__module__,
                target.__qualname__,
                self.options.execution_timeout,
                self.options.worker_start_method,
                self.options.memory_limit_mb,
                self.options.max_output_chars,
                self.options.measure_coverage,
                self.instrumented.filename,
                self.options.coverage_sources,
                self.options.coverage_include,
                self.options.coverage_omit,
                self.options.instrument_packages,
                self.options.instrument_include,
                self.options.instrument_omit,
                self.options.instrument_import_submodules,
            )

    def trace(self, inputs: Mapping[str, Any]) -> ExecutionTrace:
        """Execute one synchronous input and return its validated symbolic trace."""
        if inspect.iscoroutinefunction(self.target):
            raise TypeError("async targets require await engine.atrace(inputs)")
        if self._isolated_executor is not None:
            return self._isolated_executor.trace(inputs)
        if self.options.measure_coverage:
            coverage_session = self._new_coverage_session()
            coverage_session.start()
            try:
                trace = self._trace_sync(dict(inputs))
            finally:
                coverage_data = coverage_session.stop()
            trace.coverage_data = coverage_data
            return trace
        return self._trace_sync(dict(inputs))

    async def atrace(self, inputs: Mapping[str, Any]) -> ExecutionTrace:
        """Execute one synchronous or asynchronous target in the current task."""
        if self._isolated_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(self._isolated_executor.trace, inputs)
            )
        concrete_inputs = copy.deepcopy(dict(inputs))
        recorded_inputs = copy.deepcopy(concrete_inputs)
        self.instrumented.refresh_environment()
        args, kwargs = self._call_arguments(concrete_inputs)
        started = time.perf_counter()
        coverage_session = (
            self._new_coverage_session() if self.options.measure_coverage else None
        )
        if coverage_session is not None:
            coverage_session.start()
        try:
            with session_scope(concrete_inputs) as runtime_session:
                try:
                    with self._instrumentation_context():
                        value = self.instrumented.callable(*args, **kwargs)
                        if inspect.isawaitable(value):
                            value = await value
                        if inspect.isasyncgen(value):
                            value = await self._consume_async_generator(
                                value, runtime_session
                            )
                        elif inspect.isgenerator(value):
                            value = self._consume_generator(value, runtime_session)
                    outcome = ExecutionOutcome.RETURNED
                    exception_type = None
                    exception_message = None
                except (Exception, SystemExit) as exc:
                    value = None
                    outcome = ExecutionOutcome.RAISED
                    exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                    exception_message = str(exc)
                trace = ExecutionTrace(
                    inputs=recorded_inputs,
                    outcome=outcome,
                    branches=list(runtime_session.branches),
                    diagnostics=list(runtime_session.diagnostics),
                    return_value=value,
                    exception_type=exception_type,
                    exception_message=exception_message,
                    elapsed_seconds=time.perf_counter() - started,
                    input_symbols=dict(runtime_session.input_symbols),
                    input_paths=dict(runtime_session.input_paths),
                )
        finally:
            coverage_data = (
                coverage_session.stop() if coverage_session is not None else None
            )
        trace.coverage_data = coverage_data
        return trace

    def explore(self, seeds: Sequence[Mapping[str, Any]]) -> ConcolicResult:
        """Explore a synchronous target from one or more concrete seeds."""
        if not seeds:
            raise ValueError("at least one concrete seed is required")
        if inspect.iscoroutinefunction(self.target):
            raise TypeError("async targets require await engine.aexplore(seeds)")
        return self._explore_sync(self._prepare_seeds(seeds))

    async def aexplore(self, seeds: Sequence[Mapping[str, Any]]) -> ConcolicResult:
        """Explore a synchronous or asynchronous target."""
        if not seeds:
            raise ValueError("at least one concrete seed is required")
        return await self._explore_async(self._prepare_seeds(seeds))

    def shrink(self, trace: ExecutionTrace):
        """Minimize a failing trace and revalidate the resulting input."""
        from .shrink import CounterexampleShrinker

        if inspect.iscoroutinefunction(self.target):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(CounterexampleShrinker(self).ashrink(trace))
            raise RuntimeError("await engine.ashrink(trace) inside an event loop")
        return CounterexampleShrinker(self).shrink(trace)

    async def ashrink(self, trace: ExecutionTrace):
        """Asynchronously minimize and revalidate a failing trace."""
        from .shrink import CounterexampleShrinker

        return await CounterexampleShrinker(self).ashrink(trace)

    def _trace_sync(self, inputs: Dict[str, Any]) -> ExecutionTrace:
        inputs = copy.deepcopy(inputs)
        recorded_inputs = copy.deepcopy(inputs)
        self.instrumented.refresh_environment()
        args, kwargs = self._call_arguments(inputs)
        started = time.perf_counter()
        with session_scope(inputs) as runtime_session:
            try:
                with self._instrumentation_context():
                    value = self.instrumented.callable(*args, **kwargs)
                    if inspect.isawaitable(value):
                        raise TypeError("target returned an awaitable; use atrace()")
                    if inspect.isgenerator(value):
                        value = self._consume_generator(value, runtime_session)
                outcome = ExecutionOutcome.RETURNED
                exception_type = None
                exception_message = None
            except (Exception, SystemExit) as exc:
                value = None
                outcome = ExecutionOutcome.RAISED
                exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                exception_message = str(exc)
            return ExecutionTrace(
                inputs=recorded_inputs,
                outcome=outcome,
                branches=list(runtime_session.branches),
                diagnostics=list(runtime_session.diagnostics),
                return_value=value,
                exception_type=exception_type,
                exception_message=exception_message,
                elapsed_seconds=time.perf_counter() - started,
                input_symbols=dict(runtime_session.input_symbols),
                input_paths=dict(runtime_session.input_paths),
            )

    def _consume_generator(self, generator, runtime_session) -> List[Any]:
        values = []
        for _ in range(self.options.max_yields):
            try:
                values.append(next(generator))
            except StopIteration:
                return values
        generator.close()
        self._generator_truncated(runtime_session)
        return values

    async def _consume_async_generator(self, generator, runtime_session) -> List[Any]:
        values = []
        for _ in range(self.options.max_yields):
            try:
                values.append(await generator.__anext__())
            except StopAsyncIteration:
                return values
        await generator.aclose()
        self._generator_truncated(runtime_session)
        return values

    def _generator_truncated(self, runtime_session) -> None:
        from .models import DiagnosticCode

        runtime_session.diagnose(
            DiagnosticCode.GENERATOR_TRUNCATED,
            f"generator consumption stopped after {self.options.max_yields} values",
        )

    def _explore_sync(self, seeds: Sequence[Mapping[str, Any]]) -> ConcolicResult:
        return self._explore_core(seeds, self.trace)

    async def _explore_async(
        self, seeds: Sequence[Mapping[str, Any]]
    ) -> ConcolicResult:
        traces: List[ExecutionTrace] = []
        generated: List[Dict[str, Any]] = []
        pending_inputs: Deque[Dict[str, Any]] = deque(dict(seed) for seed in seeds)
        frontiers: Deque[Frontier] = deque()
        seen_inputs = set()
        seen_frontiers = set()
        covered_edges = set()
        campaign_diagnostics: List[Diagnostic] = []
        branch_frequency: Counter[str] = Counter()
        path_counts: Counter[str] = Counter()
        search_stats = self._new_search_stats()
        deadline = self._campaign_deadline()
        while len(traces) < self.options.max_iterations and (
            pending_inputs or frontiers
        ):
            if self._campaign_expired(deadline, campaign_diagnostics, search_stats):
                break
            if not pending_inputs:
                self._sample_frontier(
                    frontiers,
                    pending_inputs,
                    seen_inputs,
                    campaign_diagnostics,
                )
                if not pending_inputs:
                    continue
            inputs = pending_inputs.popleft()
            key = _input_key(inputs)
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            trace = await self.atrace(inputs)
            traces.append(trace)
            generated.append(inputs)
            covered_edges.update(trace.covered_edges)
            branch_frequency.update(branch.branch_id for branch in trace.branches)
            path_state = _path_state_key(trace)
            path_counts[path_state] += 1
            if path_counts[path_state] <= self.options.max_inputs_per_path:
                self._enqueue_frontiers(
                    trace,
                    frontiers,
                    seen_frontiers,
                    covered_edges,
                    branch_frequency,
                    search_stats,
                )
            else:
                search_stats["subsumed_paths"] += 1
            if self.options.stop_on_error and trace.outcome == ExecutionOutcome.RAISED:
                break
        result = ConcolicResult(
            traces,
            generated,
            exhausted=not pending_inputs and not frontiers,
            iterations=len(traces),
            pending_frontiers=len(frontiers),
            diagnostics=campaign_diagnostics,
            search_stats=self._finalize_search_stats(search_stats),
        )
        self._attach_coverage(result)
        if self.options.explain_gaps:
            result.gaps = explain_gaps(result)
        return result

    def _explore_core(
        self,
        seeds: Sequence[Mapping[str, Any]],
        trace_function: Callable[[Mapping[str, Any]], ExecutionTrace],
    ) -> ConcolicResult:
        traces: List[ExecutionTrace] = []
        generated: List[Dict[str, Any]] = []
        pending_inputs: Deque[Dict[str, Any]] = deque(dict(seed) for seed in seeds)
        frontiers: Deque[Frontier] = deque()
        seen_inputs = set()
        seen_frontiers = set()
        covered_edges = set()
        campaign_diagnostics: List[Diagnostic] = []
        branch_frequency: Counter[str] = Counter()
        path_counts: Counter[str] = Counter()
        search_stats = self._new_search_stats()
        deadline = self._campaign_deadline()
        while len(traces) < self.options.max_iterations and (
            pending_inputs or frontiers
        ):
            if self._campaign_expired(deadline, campaign_diagnostics, search_stats):
                break
            if not pending_inputs:
                self._sample_frontier(
                    frontiers,
                    pending_inputs,
                    seen_inputs,
                    campaign_diagnostics,
                )
                if not pending_inputs:
                    continue
            inputs = pending_inputs.popleft()
            key = _input_key(inputs)
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            trace = trace_function(inputs)
            traces.append(trace)
            generated.append(inputs)
            covered_edges.update(trace.covered_edges)
            branch_frequency.update(branch.branch_id for branch in trace.branches)
            path_state = _path_state_key(trace)
            path_counts[path_state] += 1
            if path_counts[path_state] <= self.options.max_inputs_per_path:
                self._enqueue_frontiers(
                    trace,
                    frontiers,
                    seen_frontiers,
                    covered_edges,
                    branch_frequency,
                    search_stats,
                )
            else:
                search_stats["subsumed_paths"] += 1
            if self.options.stop_on_error and trace.outcome == ExecutionOutcome.RAISED:
                break
        result = ConcolicResult(
            traces,
            generated,
            exhausted=not pending_inputs and not frontiers,
            iterations=len(traces),
            pending_frontiers=len(frontiers),
            diagnostics=campaign_diagnostics,
            search_stats=self._finalize_search_stats(search_stats),
        )
        self._attach_coverage(result)
        if self.options.explain_gaps:
            result.gaps = explain_gaps(result)
        return result

    def _enqueue_frontiers(
        self,
        trace: ExecutionTrace,
        queue: Deque[Frontier],
        seen: set[str],
        covered_edges: set[tuple[str, bool]],
        branch_frequency: Counter[str],
        search_stats: Dict[str, Any],
    ) -> None:
        for depth, branch in enumerate(trace.branches):
            if depth >= self.options.max_path_depth:
                search_stats["pruned_depth"] += 1
                continue
            formula = branch.frontier_formula()
            if formula is None:
                continue
            if self.options.input_constraint is not None:
                formula = z3.And(
                    self.options.input_constraint(trace.input_symbols), formula
                )
            formula = z3.simplify(formula)
            if _z3_node_count(formula) > self.options.max_constraint_nodes:
                search_stats["pruned_constraint_size"] += 1
                continue
            frontier = Frontier(
                branch.branch_id,
                not branch.taken,
                formula,
                dict(trace.input_symbols),
                dict(trace.input_paths),
                dict(trace.inputs),
                depth,
            )
            if frontier.signature in seen or len(seen) >= self.options.max_frontiers:
                search_stats["duplicate_or_capped_frontiers"] += 1
                continue
            seen.add(frontier.signature)
            search_stats["enqueued_frontiers"] += 1
            if self.options.strategy == ExplorationStrategy.DEPTH_FIRST:
                queue.appendleft(frontier)
            elif (
                self.options.strategy == ExplorationStrategy.COVERAGE
                and (frontier.branch_id, frontier.desired_taken) not in covered_edges
            ):
                queue.append(frontier)
                self._order_coverage_frontiers(
                    queue,
                    covered_edges,
                    branch_frequency,
                )
            else:
                queue.append(frontier)

    def _sample_frontier(
        self,
        frontiers: Deque[Frontier],
        pending_inputs: Deque[Dict[str, Any]],
        seen_inputs: set[Any],
        campaign_diagnostics: List[Diagnostic],
    ) -> None:
        if not frontiers:
            return
        frontier = frontiers.popleft()
        batch = self.sampler.sample(
            frontier.formula,
            frontier.input_symbols,
            self.options.samples_per_frontier,
            self.options.random_seed,
            self.options.solver_timeout,
        )
        campaign_diagnostics.extend(batch.diagnostics)
        for candidate in batch.candidates:
            complete_candidate = materialize_candidate(
                frontier.base_inputs,
                candidate,
                frontier.input_paths,
            )
            if _input_key(complete_candidate) not in seen_inputs:
                pending_inputs.append(complete_candidate)

    def _campaign_deadline(self) -> Optional[float]:
        if self.options.campaign_timeout is None:
            return None
        return time.monotonic() + self.options.campaign_timeout

    def _instrumentation_context(self):
        if self.package_instrumentor is None:
            return nullcontext()
        return self.package_instrumentor.activated()

    def _prepare_seeds(
        self, seeds: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        if not self.options.expand_input_shapes:
            return seeds
        return expand_input_shapes(seeds, self.options.max_shape_variants)

    @staticmethod
    def _new_search_stats() -> Dict[str, Any]:
        return {
            "enqueued_frontiers": 0,
            "duplicate_or_capped_frontiers": 0,
            "pruned_depth": 0,
            "pruned_constraint_size": 0,
            "subsumed_paths": 0,
            "campaign_timed_out": False,
        }

    def _campaign_expired(
        self,
        deadline: Optional[float],
        diagnostics: List[Diagnostic],
        stats: Dict[str, Any],
    ) -> bool:
        if deadline is None or time.monotonic() < deadline:
            return False
        if not stats["campaign_timed_out"]:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.CAMPAIGN_TIMEOUT,
                    f"campaign exceeded its {self.options.campaign_timeout:g}s limit",
                )
            )
            stats["campaign_timed_out"] = True
        return True

    def _finalize_search_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(stats)
        cache_info = getattr(self.sampler, "cache_info", None)
        if cache_info is not None:
            result["sampler_cache"] = cache_info
        if self.package_instrumentor is not None:
            result["instrumentation"] = self.package_instrumentor.report.to_dict()
        return result

    def coverage_report(self, traces: Sequence[ExecutionTrace]):
        """Merge conventional coverage from existing direct or worker traces."""
        accumulator = CoverageAccumulator(
            self.instrumented.filename,
            self.target.__qualname__,
            self.options.coverage_sources,
            self.options.coverage_include,
            self.options.coverage_omit,
        )
        for trace in traces:
            accumulator.add(trace.coverage_data)
        return accumulator.report()

    def _attach_coverage(self, result: ConcolicResult) -> None:
        if not self.options.measure_coverage:
            return
        try:
            accumulator = self._coverage_accumulator()
            timeline = []
            for iteration, trace in enumerate(result.traces, 1):
                accumulator.add(trace.coverage_data)
                report = accumulator.report()
                timeline.append(
                    CoverageSnapshot(
                        iteration=iteration,
                        lines=report.lines.percent,
                        statements=report.statements.percent,
                        functions=report.functions.percent,
                        branches=report.branches.percent,
                    )
                )
            result.coverage = timeline and report or accumulator.report()
            result.coverage_timeline = tuple(timeline)
            result.coverage_gate = self._coverage_gate(result.coverage)
        except Exception as exc:
            result.diagnostics.append(
                Diagnostic(
                    DiagnosticCode.COVERAGE_FAILURE,
                    f"could not calculate conventional coverage: {exc}",
                )
            )

    def _coverage_accumulator(self) -> CoverageAccumulator:
        return CoverageAccumulator(
            self.instrumented.filename,
            self.target.__qualname__,
            self.options.coverage_sources,
            self.options.coverage_include,
            self.options.coverage_omit,
        )

    def _coverage_gate(self, report) -> Optional[CoverageGate]:
        required = self.options.coverage_thresholds
        if not required:
            return None
        actual = {
            "lines": report.lines.percent,
            "statements": report.statements.percent,
            "functions": report.functions.percent,
            "branches": report.branches.percent,
        }
        failures = tuple(
            f"{name}: {actual[name]:.2f}% < {minimum:.2f}%"
            for name, minimum in required.items()
            if actual[name] < minimum
        )
        return CoverageGate(
            passed=not failures,
            required=required,
            actual=actual,
            failures=failures,
        )

    def _new_coverage_session(self) -> CoverageSession:
        return CoverageSession(
            self.instrumented.filename,
            self.options.coverage_sources,
            self.options.coverage_include,
            self.options.coverage_omit,
        )

    @staticmethod
    def _order_coverage_frontiers(
        queue: Deque[Frontier],
        covered_edges: set[tuple[str, bool]],
        branch_frequency: Counter[str],
    ) -> None:
        ordered = sorted(
            queue,
            key=lambda frontier: (
                (frontier.branch_id, frontier.desired_taken) in covered_edges,
                branch_frequency[frontier.branch_id],
                _z3_node_count(frontier.formula),
                frontier.depth,
            ),
        )
        queue.clear()
        queue.extend(ordered)

    def _call_arguments(
        self, inputs: Mapping[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        return build_call_arguments(self.signature, inputs)


def _input_key(inputs: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (name, type(value).__qualname__, repr(value))
            for name, value in inputs.items()
        )
    )


def _path_state_key(trace: ExecutionTrace) -> str:
    return "|".join(
        (
            trace.path_signature,
            trace.outcome.value,
            trace.exception_type or "",
        )
    )


def _z3_node_count(expression: z3.ExprRef) -> int:
    seen = set()
    stack = [expression]
    while stack:
        current = stack.pop()
        identifier = current.get_id()
        if identifier in seen:
            continue
        seen.add(identifier)
        stack.extend(current.children())
    return len(seen)


ConcolicEngine = NativeConcolicEngine
