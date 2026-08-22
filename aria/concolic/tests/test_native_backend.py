"""End-to-end tests for native source instrumentation and exploration."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from posixpath import basename as path_basename
from posixpath import join as path_join
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import z3
import pytest

from aria.concolic import (
    ArtifactStore,
    ConcolicEngine,
    ConcolicOptions,
    ConcolicResult,
    DiagnosticCode,
    Diagnostic,
    ExecutionOutcome,
    SymbolicModelRegistry,
    available_model_packs,
    assert_concolic,
    ConcolicAssertionError,
    hypothesis_strategy,
    pytest_parameters,
    register_function_model,
)
from aria.cli.concolic_cli import main as concolic_main
from aria.concolic.sampling import CandidateBatch, ConstraintSampler
from aria.concolic.models import SymbolicTerm
from aria.concolic.runtime import RuntimeSession
from aria.concolic.coverage import _coverage_report_from_json
from aria.concolic.symbolic import exact_string_val
from aria.concolic.tests.fixtures.interpkg.api import entry as package_entry
from aria.concolic.tests.fixtures.interpkg.api import (
    expansion_entry,
    lazy_entry,
    property_entry,
)
from aria.concolic.tests.fixtures.interpkg.core import PropertyBox


def arithmetic_target(x: int, y: int = 2) -> str:
    total = x + y
    if total > 10:
        return "large"
    return "small"


def nested_target(x: int) -> str:
    y = x * 2
    if y >= 0:
        if y < 8:
            return "middle"
        return "high"
    return "negative"


def failure_target(x: int) -> int:
    if x == 7:
        raise ValueError("seven is forbidden")
    return 100 // (x - 1)


def string_target(value: str) -> str:
    if len(value) >= 3 and value.startswith("ab"):
        return "matched"
    return "other"


def negative_index_target(value: str) -> bool:
    if value[-1] == "x":
        return True
    return False


def division_target(x: int, y: int) -> int:
    quotient = x // y
    remainder = x % y
    if quotient == 2 and remainder == -1:
        return 1
    return 0


async def async_target(x: int) -> int:
    await asyncio.sleep(0)
    if x > 5:
        return 1
    return 0


async def async_failure_target(x: int) -> int:
    await asyncio.sleep(0)
    if x >= 10:
        raise RuntimeError("async too large")
    return x


_side_effect_calls = 0


def _side_effect(value: int) -> int:
    global _side_effect_calls
    _side_effect_calls += 1
    return value


def side_effect_target(x: int) -> int:
    if _side_effect(x) > 0:
        return 1
    return 0


def short_circuit_target(x: int) -> int:
    if x < 0 and _side_effect(x) == -1:
        return 1
    return 0


class OpaquePayload:
    __slots__ = ("size",)

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other) -> bool:
        return isinstance(other, OpaquePayload) and self.size == other.size


def unsupported_input_target(items: OpaquePayload) -> int:
    if len(items) > 1:
        return 1
    return 0


def slow_target(delay: int) -> int:
    time.sleep(delay)
    return delay


def campaign_slow_target(value: int) -> int:
    time.sleep(0.03)
    if value > 0:
        return 1
    return 0


def shrink_target(x: int) -> int:
    if x >= 10:
        raise RuntimeError("too large")
    return x


def range_target(limit: int) -> int:
    total = 0
    for index in range(limit):
        total += index
        if index == 2:
            return total
    return -1


def generator_target(limit: int):
    for index in range(limit):
        if index >= 2:
            yield index


def mixed_input_target(x: int, payload: OpaquePayload) -> int:
    if x > 0:
        return len(payload)
    return -len(payload)


def structured_target(
    values: list[int], config: dict[str, int], pair: tuple[int, int]
) -> bool:
    total = values[0] + values[-1] + pair[0]
    if total > config["limit"]:
        return True
    return False


def dynamic_list_target(values: list[int], index: int) -> bool:
    if values[index] > 5:
        return True
    return False


def structured_failure_target(values: list[int]) -> int:
    if values[0] >= 10:
        raise RuntimeError("structured too large")
    return values[0]


def list_mutation_target(values: list[int], added: int) -> bool:
    values.append(added)
    values[0] = values[0] + 1
    removed = values.pop()
    if len(values) == 1 and values[0] > 5 and removed > 10:
        return True
    return False


def dict_mutation_target(values: dict[str, int], delta: int) -> bool:
    values["score"] = values["score"] + delta
    removed = values.pop("bonus")
    values.setdefault("restored", removed)
    if values["score"] > 10 and values["restored"] > 3:
        return True
    return False


@dataclass
class Account:
    balance: int
    limit: int
    label: str


def object_mutation_target(account: Account, delta: int) -> bool:
    account.balance += delta
    account.label = account.label + "!"
    if account.balance > account.limit and account.label.startswith("vip"):
        return True
    return False


def parsing_pack_target(text: str) -> bool:
    if int(text) > 10:
        return True
    return False


def collections_pack_target(values: list[int]) -> bool:
    if sum(values) > 10 and max(values) > 5:
        return True
    return False


def text_pack_target(code: int) -> bool:
    if ord(chr(code)) == 66:
        return True
    return False


def path_pack_target(root: str, name: str) -> bool:
    combined = path_join(root, name)
    leaf = path_basename(name)
    if combined.endswith("/" + leaf):
        return True
    return False


def url_string_model_target(text: str) -> bool:
    cleaned = text.lstrip("\x00 ").replace("\t", "")
    position = cleaned.find(":")
    if position > 0 and cleaned[:position].isascii():
        return True
    return False


def split_model_target(text: str) -> bool:
    if "#" in text:
        before, after = text.split("#", 1)
        if before == "path" and len(after) > 0:
            return True
    return False


def buggy_absolute(value: int) -> int:
    if value < 0:
        return value
    return value


def bytes_target(data: bytes) -> bool:
    if len(data) >= 2 and data.startswith(b"AB"):
        return True
    return False


def bytearray_target(data: bytearray, added: int) -> bool:
    data.append(added)
    if data[-1] == 65:
        return True
    return False


def set_target(values: set[int], needle: int) -> bool:
    if needle in values:
        return True
    return False


def float_target(value: float) -> bool:
    scaled = value / 2.0
    if scaled > 3.5:
        return True
    return False


def shape_target(values: list[int]) -> str:
    if len(values) == 0:
        return "empty"
    if len(values) >= 3:
        return "large"
    return "small"


def clamp_positive(value: int) -> int:
    return value if value >= 0 else 0


def custom_model_target(value: int) -> bool:
    if clamp_positive(value) > 5:
        return True
    return False


def noisy_target(x: int) -> int:
    print("x" * 1_000)
    return x


def system_exit_target(code: int) -> None:
    sys.exit(code)


_global_limit = 3


def global_target(x: int) -> bool:
    return x > _global_limit if x >= 0 else False


def make_closure_target(limit: int):
    def closure_target(x: int) -> bool:
        if x > limit:
            return True
        return False

    return closure_target


def recursive_target(value: int) -> int:
    if value <= 0:
        return 0
    return 1 + recursive_target(value - 1)


def signature_target(x: int, /, y: int = 2, *, z: int = 3) -> bool:
    if x + y > z:
        return True
    return False


def positional_failure_target(value: int, /, *, threshold: int = 10) -> int:
    if value >= threshold:
        raise ValueError("positional failure")
    return value


class MethodTarget:
    def classify(self, value: int) -> bool:
        if value > 4:
            return True
        return False


class TestTracing:
    def test_assignment_and_branch_are_symbolically_validated(self) -> None:
        trace = ConcolicEngine(arithmetic_target).trace({"x": 1})

        assert trace.outcome == ExecutionOutcome.RETURNED
        assert trace.return_value == "small"
        assert len(trace.branches) == 1
        assert trace.branches[0].taken is False
        assert trace.branches[0].predicate is not None
        assert "x" in trace.input_symbols
        assert "y" not in trace.input_symbols
        assert trace.is_sound

    def test_nested_branches_preserve_prefix_constraints(self) -> None:
        trace = ConcolicEngine(nested_target).trace({"x": 1})

        assert [branch.taken for branch in trace.branches] == [True, True]
        assert len(trace.branches[1].path_prefix) == 1
        assert all(branch.predicate is not None for branch in trace.branches)

    def test_python_floor_division_and_modulo_match_negative_divisor(self) -> None:
        trace = ConcolicEngine(division_target).trace({"x": -7, "y": -3})

        assert trace.return_value == 1
        assert trace.is_sound
        assert trace.branches[0].predicate is not None

    def test_string_length_and_prefix_models(self) -> None:
        trace = ConcolicEngine(string_target).trace({"value": "abc"})

        assert trace.return_value == "matched"
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_negative_string_index_uses_python_normalization(self) -> None:
        trace = ConcolicEngine(negative_index_target).trace({"value": "ax"})

        assert trace.return_value is True
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_false_definedness_guard_is_not_accepted_as_equality(self) -> None:
        session = RuntimeSession({"x": 0})
        session.enter("guard-test", {"x": 0}, ["x"])
        symbol = session.input_symbols["x"]

        valid = session.validate(SymbolicTerm(symbol, (symbol > 0,)), 0)

        assert valid is False
        assert any(
            diagnostic.code == DiagnosticCode.DEFINEDNESS_MISMATCH
            for diagnostic in session.diagnostics
        )

    def test_target_expression_is_evaluated_only_once(self) -> None:
        global _side_effect_calls
        _side_effect_calls = 0

        trace = ConcolicEngine(side_effect_target).trace({"x": 1})

        assert trace.return_value == 1
        assert _side_effect_calls == 1
        assert trace.branches[0].predicate is None
        assert any(
            diagnostic.code == DiagnosticCode.UNSUPPORTED_EXPRESSION
            for diagnostic in trace.diagnostics
        )

    def test_symbolic_evaluation_respects_concrete_short_circuiting(self) -> None:
        global _side_effect_calls
        _side_effect_calls = 0

        trace = ConcolicEngine(short_circuit_target).trace({"x": 2})

        assert trace.return_value == 0
        assert _side_effect_calls == 0
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_unsupported_inputs_are_explicitly_diagnosed(self) -> None:
        trace = ConcolicEngine(unsupported_input_target).trace(
            {"items": OpaquePayload(2)}
        )

        assert trace.return_value == 1
        assert any(
            diagnostic.code == DiagnosticCode.UNSUPPORTED_INPUT
            for diagnostic in trace.diagnostics
        )
        assert trace.branches[0].predicate is None

    def test_program_exception_is_an_outcome_not_an_engine_failure(self) -> None:
        trace = ConcolicEngine(failure_target).trace({"x": 7})

        assert trace.outcome == ExecutionOutcome.RAISED
        assert trace.exception_type == "builtins.ValueError"
        assert trace.exception_message == "seven is forbidden"

    def test_system_exit_is_recorded_without_terminating_runner(self) -> None:
        trace = ConcolicEngine(system_exit_target).trace({"code": 7})

        assert trace.outcome == ExecutionOutcome.RAISED
        assert trace.exception_type == "builtins.SystemExit"
        assert trace.exception_message == "7"

    def test_range_loop_records_each_iteration_and_exit(self) -> None:
        trace = ConcolicEngine(range_target).trace({"limit": 2})

        assert trace.return_value == -1
        loop_id = trace.branches[0].branch_id
        loop_observations = [
            branch for branch in trace.branches if branch.branch_id == loop_id
        ]
        assert [branch.taken for branch in loop_observations] == [True, True, False]
        assert all(branch.predicate is not None for branch in loop_observations)
        assert trace.is_sound

    def test_generator_is_driven_under_the_same_runtime_session(self) -> None:
        trace = ConcolicEngine(generator_target).trace({"limit": 4})

        assert trace.return_value == [2, 3]
        assert trace.branches
        assert trace.is_sound

    def test_primitive_global_is_refreshed_without_losing_symbolic_precision(
        self,
    ) -> None:
        global _global_limit
        original_limit = _global_limit
        engine = ConcolicEngine(global_target)
        try:
            _global_limit = 10
            trace = engine.trace({"x": 5})
        finally:
            _global_limit = original_limit

        assert trace.return_value is False
        assert all(branch.predicate is not None for branch in trace.branches)
        assert trace.is_sound

    def test_primitive_closure_read_is_modeled_in_direct_mode(self) -> None:
        engine = ConcolicEngine(make_closure_target(4))

        trace = engine.trace({"x": 5})

        assert trace.return_value is True
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_self_recursion_propagates_symbolic_arguments(self) -> None:
        trace = ConcolicEngine(recursive_target).trace({"value": 3})

        assert trace.return_value == 3
        assert [branch.taken for branch in trace.branches] == [
            False,
            False,
            False,
            True,
        ]
        assert all(branch.predicate is not None for branch in trace.branches)
        assert not any(
            diagnostic.code == DiagnosticCode.NESTED_SYMBOLIC_FRAME
            for diagnostic in trace.diagnostics
        )
        assert trace.is_sound

    def test_positional_only_and_keyword_only_parameters_preserve_signature(
        self,
    ) -> None:
        trace = ConcolicEngine(signature_target).trace({"x": 2, "z": 5})

        assert trace.return_value is False
        assert trace.branches[0].predicate is not None
        assert set(trace.input_symbols) == {"x", "z"}
        assert trace.is_sound

    def test_bound_method_runs_in_direct_mode(self) -> None:
        trace = ConcolicEngine(MethodTarget().classify).trace({"value": 5})

        assert trace.return_value is True
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_branch_identifiers_and_locations_are_stable(self) -> None:
        first = ConcolicEngine(arithmetic_target).trace({"x": 1})
        second = ConcolicEngine(arithmetic_target).trace({"x": 20})

        assert first.branches[0].branch_id == second.branches[0].branch_id
        assert first.branches[0].location == second.branches[0].location

    def test_custom_function_model_preserves_symbolic_call(self) -> None:
        def clamp_model(args):
            if len(args) != 1 or args[0].expression is None:
                return None
            expression = args[0].expression
            return SymbolicTerm(
                z3.If(expression >= 0, expression, 0),
                args[0].guards,
            )

        register_function_model("clamp_positive", clamp_model, replace=True)

        trace = ConcolicEngine(custom_model_target).trace({"value": 9})

        assert trace.return_value is True
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_model_registry_clones_function_and_method_extensions(self) -> None:
        registry = SymbolicModelRegistry()
        registry.register_function("identity", lambda args: args[0])
        registry.register_method("identity", lambda owner, args: owner)

        cloned = registry.clone()

        assert cloned.function_names == ("identity",)
        assert cloned.method_names == ("identity",)
        assert cloned.function_model("identity") is not None
        assert cloned.method_model("identity") is not None


class TestExploration:
    def test_sampling_reaches_both_edges(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(max_iterations=8, samples_per_frontier=3, random_seed=4),
        )

        result = engine.explore([{"x": 1}])

        outcomes = {trace.return_value for trace in result.traces}
        assert outcomes == {"small", "large"}
        assert len(result.covered_edges) == 2

    def test_input_constraint_is_applied_to_generated_inputs(self) -> None:
        engine = ConcolicEngine(
            nested_target,
            ConcolicOptions(
                max_iterations=10,
                samples_per_frontier=3,
                input_constraint=lambda symbols: z3.And(
                    symbols["x"] >= -3, symbols["x"] <= 3
                ),
            ),
        )

        result = engine.explore([{"x": 1}])

        assert result.generated_inputs
        assert all(-3 <= inputs["x"] <= 3 for inputs in result.generated_inputs)

    def test_runtime_state_is_isolated_between_threads(self) -> None:
        engine = ConcolicEngine(arithmetic_target)

        with ThreadPoolExecutor(max_workers=4) as executor:
            traces = list(
                executor.map(lambda value: engine.trace({"x": value}), range(8))
            )

        assert [trace.inputs["x"] for trace in traces] == list(range(8))
        assert all(len(trace.branches) == 1 for trace in traces)
        assert all(trace.is_sound for trace in traces)

    def test_range_frontiers_grow_loop_until_inner_branch_is_reached(self) -> None:
        engine = ConcolicEngine(
            range_target,
            ConcolicOptions(max_iterations=20, samples_per_frontier=2),
        )

        result = engine.explore([{"limit": 0}])

        assert any(trace.return_value == 3 for trace in result.traces)

    def test_opaque_seed_components_are_preserved_during_sampling(self) -> None:
        engine = ConcolicEngine(mixed_input_target, ConcolicOptions(max_iterations=4))

        result = engine.explore([{"x": 0, "payload": OpaquePayload(3)}])

        assert {trace.return_value for trace in result.traces} == {-3, 3}
        assert all(
            inputs["payload"] == OpaquePayload(3) for inputs in result.generated_inputs
        )

    def test_fixed_shape_structures_are_reconstructed_from_samples(self) -> None:
        engine = ConcolicEngine(
            structured_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        )
        seed = {"values": [1, 2], "config": {"limit": 20}, "pair": (3, 4)}

        result = engine.explore([seed])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(len(item["values"]) == 2 for item in result.generated_inputs)
        assert all(set(item["config"]) == {"limit"} for item in result.generated_inputs)
        assert all(isinstance(item["pair"], tuple) for item in result.generated_inputs)

    def test_symbolic_dynamic_list_index_stays_within_shape(self) -> None:
        engine = ConcolicEngine(
            dynamic_list_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        )

        result = engine.explore([{"values": [1, 9], "index": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(
            -len(inputs["values"]) <= inputs["index"] < len(inputs["values"])
            for inputs in result.generated_inputs
        )

    def test_slia_sampling_reaches_string_prefix_branch(self) -> None:
        engine = ConcolicEngine(
            string_target,
            ConcolicOptions(max_iterations=20, samples_per_frontier=4),
        )

        result = engine.explore([{"value": ""}])

        assert any(trace.return_value == "matched" for trace in result.traces)

    def test_sampler_diagnostics_are_retained_by_campaign(self) -> None:
        class DiagnosticSampler:
            def sample(self, *args, **kwargs):
                return CandidateBatch(
                    [],
                    [
                        Diagnostic(
                            DiagnosticCode.SAMPLER_FAILURE,
                            "intentional sampler diagnostic",
                        )
                    ],
                    {},
                )

        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(max_iterations=2),
            sampler=DiagnosticSampler(),
        )

        result = engine.explore([{"x": 1}])

        assert result.diagnostics[0].code == DiagnosticCode.SAMPLER_FAILURE

    def test_constraint_sampler_reuses_bounded_formula_cache(self) -> None:
        sampler = ConstraintSampler(cache_size=2)
        value = z3.Int("cache_value")
        formula = value > 3

        first = sampler.sample(formula, {"cache_value": value}, 2, 7, 1.0)
        second = sampler.sample(formula, {"cache_value": value}, 2, 7, 1.0)

        assert first.candidates == second.candidates
        assert second.stats["cache_hit"] is True
        assert sampler.cache_info["hits"] == 1

    def test_constraint_size_and_depth_budgets_prune_frontiers(self) -> None:
        size_limited = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(max_iterations=5, max_constraint_nodes=1),
        ).explore([{"x": 1}])
        depth_limited = ConcolicEngine(
            nested_target,
            ConcolicOptions(max_iterations=5, max_path_depth=1),
        ).explore([{"x": 1}])

        assert size_limited.search_stats["pruned_constraint_size"] >= 1
        assert depth_limited.search_stats["pruned_depth"] >= 1

    def test_path_subsumption_limits_reexpansion(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(
                max_iterations=8,
                samples_per_frontier=6,
                max_inputs_per_path=1,
            ),
        )

        result = engine.explore([{"x": 1}])

        assert result.search_stats["subsumed_paths"] >= 1

    def test_campaign_deadline_stops_between_executions(self) -> None:
        engine = ConcolicEngine(
            campaign_slow_target,
            ConcolicOptions(max_iterations=10, campaign_timeout=0.01),
        )

        result = engine.explore([{"value": 0}])

        assert result.search_stats["campaign_timed_out"] is True
        assert any(
            diagnostic.code == DiagnosticCode.CAMPAIGN_TIMEOUT
            for diagnostic in result.diagnostics
        )


class TestWorkerIsolation:
    def test_trace_round_trips_formulas_through_spawn_worker(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(
                isolate=True, execution_timeout=5.0, worker_start_method="spawn"
            ),
        )

        trace = engine.trace({"x": 1})

        assert trace.outcome == ExecutionOutcome.RETURNED
        assert trace.return_value == "small"
        assert trace.branches[0].predicate is not None
        assert trace.is_sound

    def test_worker_is_terminated_at_wall_clock_limit(self) -> None:
        engine = ConcolicEngine(
            slow_target,
            ConcolicOptions(
                isolate=True, execution_timeout=0.1, worker_start_method="spawn"
            ),
        )

        trace = engine.trace({"delay": 2})

        assert trace.outcome == ExecutionOutcome.TIMED_OUT
        assert any(
            diagnostic.code == DiagnosticCode.EXECUTION_TIMEOUT
            for diagnostic in trace.diagnostics
        )

    def test_worker_output_capture_is_bounded(self) -> None:
        engine = ConcolicEngine(
            noisy_target,
            ConcolicOptions(
                isolate=True,
                execution_timeout=5.0,
                max_output_chars=32,
                worker_start_method="spawn",
            ),
        )

        trace = engine.trace({"x": 4})

        assert trace.return_value == 4
        assert len(trace.stdout) == 32
        assert any(
            diagnostic.code == DiagnosticCode.OUTPUT_TRUNCATED
            for diagnostic in trace.diagnostics
        )

    def test_async_target_runs_in_isolated_worker(self) -> None:
        async def run() -> None:
            engine = ConcolicEngine(
                async_target,
                ConcolicOptions(
                    isolate=True,
                    execution_timeout=5.0,
                    worker_start_method="spawn",
                ),
            )
            trace = await engine.atrace({"x": 9})

            assert trace.return_value == 1
            assert trace.branches[0].predicate is not None
            assert trace.is_sound

        asyncio.run(run())

    def test_structured_sampling_round_trips_through_workers(self) -> None:
        engine = ConcolicEngine(
            structured_target,
            ConcolicOptions(
                isolate=True,
                max_iterations=5,
                samples_per_frontier=2,
                execution_timeout=5.0,
                worker_start_method="spawn",
            ),
        )
        seed = {"values": [1, 2], "config": {"limit": 20}, "pair": (3, 4)}

        result = engine.explore([seed])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(trace.input_paths for trace in result.traces)


class TestConventionalCoverage:
    def test_direct_campaign_reports_all_conventional_percentages(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(
                measure_coverage=True,
                max_iterations=6,
                samples_per_frontier=2,
            ),
        )

        result = engine.explore([{"x": 1}])

        assert result.coverage is not None
        target = result.coverage.target_function
        assert target is not None
        assert target.lines.percent == 100.0
        assert target.statements.percent == 100.0
        assert target.functions.percent == 100.0
        assert target.branches.percent == 100.0
        assert result.coverage.functions.total > 1

    def test_partial_campaign_reports_missing_target_branches(self) -> None:
        engine = ConcolicEngine(
            nested_target,
            ConcolicOptions(measure_coverage=True, max_iterations=1),
        )

        result = engine.explore([{"x": 1}])

        assert result.coverage is not None
        target = result.coverage.target_function
        assert target is not None
        assert 0.0 < target.branches.percent < 100.0
        assert target.missing_branches

    def test_isolated_workers_merge_coverage_data(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(
                measure_coverage=True,
                isolate=True,
                max_iterations=5,
                samples_per_frontier=2,
                execution_timeout=5.0,
                worker_start_method="spawn",
            ),
        )

        result = engine.explore([{"x": 1}])

        assert result.coverage is not None
        assert result.coverage.target_function is not None
        assert result.coverage.target_function.branches.percent == 100.0

    def test_cli_writes_standalone_coverage_json(self, tmp_path: Path, capsys) -> None:
        destination = tmp_path / "coverage.json"

        status = concolic_main(
            [
                f"{__name__}:arithmetic_target",
                "--seed",
                '{"x": 1}',
                "--direct",
                "--coverage",
                "--coverage-json",
                str(destination),
                "--max-iterations",
                "5",
            ]
        )

        capsys.readouterr()
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert status == 0
        assert payload["target_function"]["branches"]["percent"] == 100.0
        assert set(payload) >= {"lines", "statements", "functions", "branches"}

    def test_function_metrics_fall_back_for_older_json_schema(self) -> None:
        lines, start = inspect.getsourcelines(arithmetic_target)
        body_lines = list(range(start + 1, start + len(lines)))
        filename = inspect.getsourcefile(arithmetic_target)
        assert filename is not None
        payload = {
            "files": {
                filename: {
                    "executed_lines": body_lines,
                    "missing_lines": [],
                    "executed_branches": [[start + 2, start + 3]],
                    "missing_branches": [],
                    "summary": {
                        "covered_lines": len(body_lines),
                        "num_statements": len(body_lines),
                        "covered_branches": 1,
                        "num_branches": 1,
                    },
                }
            },
            "totals": {
                "covered_lines": len(body_lines),
                "num_statements": len(body_lines),
                "covered_branches": 1,
                "num_branches": 1,
            },
        }

        report = _coverage_report_from_json(
            payload,
            filename,
            "arithmetic_target",
        )

        assert report.functions.total > 0
        assert report.target_function is not None

    def test_coverage_timeline_is_cumulative_and_ci_gate_is_evaluated(self) -> None:
        engine = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(
                measure_coverage=True,
                max_iterations=5,
                coverage_fail_under_branches=99.0,
            ),
        )

        result = engine.explore([{"x": 1}])

        assert len(result.coverage_timeline) == len(result.traces)
        branch_curve = [item.branches for item in result.coverage_timeline]
        assert branch_curve == sorted(branch_curve)
        assert result.coverage_gate is not None
        assert result.coverage_gate.passed is False
        assert result.coverage_gate.failures

    def test_cli_threshold_exit_and_timeline_files(
        self, tmp_path: Path, capsys
    ) -> None:
        timeline_json = tmp_path / "timeline.json"
        timeline_csv = tmp_path / "timeline.csv"

        status = concolic_main(
            [
                f"{__name__}:arithmetic_target",
                "--seed",
                '{"x": 1}',
                "--direct",
                "--max-iterations",
                "4",
                "--fail-under-functions",
                "99",
                "--coverage-timeline-json",
                str(timeline_json),
                "--coverage-timeline-csv",
                str(timeline_csv),
            ]
        )

        capsys.readouterr()
        assert status == 3
        assert json.loads(timeline_json.read_text(encoding="utf-8"))
        assert timeline_csv.read_text(encoding="utf-8").startswith(
            "iteration,lines,statements,functions,branches"
        )


class TestInterproceduralPackages:
    package_name = "aria.concolic.tests.fixtures.interpkg"

    def test_package_functions_propagate_arguments_and_returns(self) -> None:
        engine = ConcolicEngine(
            package_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                max_iterations=20,
                samples_per_frontier=4,
            ),
        )

        result = engine.explore([{"value": 0, "prefix": ""}])

        assert {trace.return_value for trace in result.traces} == {"miss", "hit"}
        assert any(len(trace.branches) >= 3 for trace in result.traces)
        instrumentation = result.search_stats["instrumentation"]
        assert any(
            name.endswith("core.classify_score")
            for name in instrumentation["functions"]
        )
        assert any(
            name.endswith("core.Formatter.decorate")
            for name in instrumentation["functions"]
        )

    def test_package_instrumentation_works_in_spawn_worker(self) -> None:
        engine = ConcolicEngine(
            package_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                isolate=True,
                execution_timeout=5.0,
                worker_start_method="spawn",
                max_iterations=16,
                samples_per_frontier=3,
            ),
        )

        result = engine.explore([{"value": 0, "prefix": ""}])

        assert {trace.return_value for trace in result.traces} == {"miss", "hit"}

    def test_package_coverage_attributes_internal_functions(self) -> None:
        engine = ConcolicEngine(
            package_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                measure_coverage=True,
                coverage_sources=(self.package_name,),
                max_iterations=16,
            ),
        )

        result = engine.explore([{"value": 0, "prefix": ""}])

        assert result.coverage is not None
        assert result.coverage.functions.covered >= 3
        assert result.coverage.branches.covered >= 6

    def test_star_args_and_star_kwargs_propagate_symbolic_values(self) -> None:
        engine = ConcolicEngine(
            expansion_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                max_iterations=16,
                samples_per_frontier=4,
            ),
        )

        result = engine.explore([{"args": [1, 2], "kwargs": {"scale": 1}}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(trace.is_sound for trace in result.traces)

    def test_instrumented_property_getter_and_setter_propagate_state(self) -> None:
        original = PropertyBox(0)
        engine = ConcolicEngine(
            property_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                max_iterations=16,
                samples_per_frontier=4,
            ),
        )

        result = engine.explore([{"box": original, "new_value": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert original.value == 0
        assert all(trace.is_sound for trace in result.traces)

    def test_package_patches_are_reentrant_and_restored(self) -> None:
        import aria.concolic.tests.fixtures.interpkg.core as core_module

        original = core_module.classify_score
        engine = ConcolicEngine(
            package_entry,
            ConcolicOptions(instrument_packages=(self.package_name,)),
        )

        assert core_module.classify_score is original
        with engine.package_instrumentor.activated():
            patched = core_module.classify_score
            assert patched is not original
            with engine.package_instrumentor.activated():
                assert core_module.classify_score is patched
        assert core_module.classify_score is original

        engine.trace({"value": 0, "prefix": ""})
        assert core_module.classify_score is original

    def test_lazy_import_hook_instruments_new_package_module(self) -> None:
        sys.modules.pop(f"{self.package_name}.lazy", None)
        engine = ConcolicEngine(
            lazy_entry,
            ConcolicOptions(
                instrument_packages=(self.package_name,),
                instrument_import_submodules=False,
                max_iterations=16,
                samples_per_frontier=4,
            ),
        )

        result = engine.explore([{"value": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert f"{self.package_name}.lazy" in engine.package_instrumentor.report.modules

    def test_multiple_package_engines_restore_shared_modules(self) -> None:
        import aria.concolic.tests.fixtures.interpkg.core as core_module

        original = core_module.classify_score
        first = ConcolicEngine(
            package_entry,
            ConcolicOptions(instrument_packages=(self.package_name,)),
        )
        second = ConcolicEngine(
            expansion_entry,
            ConcolicOptions(instrument_packages=(self.package_name,)),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            entry_trace = executor.submit(first.trace, {"value": 20, "prefix": "a"})
            expansion_trace = executor.submit(
                second.trace, {"args": [5, 6], "kwargs": {"scale": 2}}
            )

        assert entry_trace.result().return_value == "hit"
        assert expansion_trace.result().return_value is True
        assert core_module.classify_score is original


class TestContainerMutation:
    def test_list_shape_and_index_mutations_remain_symbolic(self) -> None:
        original = [0]
        engine = ConcolicEngine(
            list_mutation_target,
            ConcolicOptions(max_iterations=20, samples_per_frontier=4),
        )

        result = engine.explore([{"values": original, "added": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert original == [0]
        assert all(trace.is_sound for trace in result.traces)

    def test_dictionary_set_pop_and_setdefault_are_symbolic(self) -> None:
        seed = {"values": {"score": 0, "bonus": 1}, "delta": 0}
        engine = ConcolicEngine(
            dict_mutation_target,
            ConcolicOptions(max_iterations=20, samples_per_frontier=4),
        )

        result = engine.explore([seed])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert seed["values"] == {"score": 0, "bonus": 1}
        assert all(trace.is_sound for trace in result.traces)


class TestStatefulObjects:
    def test_dataclass_fields_are_sampled_and_mutated_symbolically(self) -> None:
        account = Account(balance=0, limit=10, label="user")
        engine = ConcolicEngine(
            object_mutation_target,
            ConcolicOptions(max_iterations=24, samples_per_frontier=5),
        )

        result = engine.explore([{"account": account, "delta": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert account == Account(balance=0, limit=10, label="user")
        assert all(
            isinstance(inputs["account"], Account) for inputs in result.generated_inputs
        )
        assert all(trace.is_sound for trace in result.traces)

    def test_dataclass_artifact_round_trip(self, tmp_path: Path) -> None:
        engine = ConcolicEngine(object_mutation_target)
        inputs = {
            "account": Account(balance=12, limit=10, label="vip"),
            "delta": 0,
        }
        trace = engine.trace(inputs)
        result = ConcolicResult(
            traces=[trace],
            generated_inputs=[inputs],
            exhausted=True,
            iterations=1,
            pending_frontiers=0,
        )
        artifact = tmp_path / "object-campaign.json"
        store = ArtifactStore()
        store.write(artifact, f"{__name__}:object_mutation_target", result)

        replay = store.replay(artifact, engine)

        assert replay.stable
        assert isinstance(replay.traces[0].inputs["account"], Account)


class TestStandardLibraryModelPacks:
    def test_default_pack_catalog(self) -> None:
        assert set(available_model_packs()) == {
            "collections",
            "numeric",
            "parsing",
            "paths",
            "text",
        }

    def test_parsing_and_collection_models_drive_exploration(self) -> None:
        parsing = ConcolicEngine(
            parsing_pack_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        ).explore([{"text": "0"}])
        collections = ConcolicEngine(
            collections_pack_target,
            ConcolicOptions(max_iterations=16, samples_per_frontier=4),
        ).explore([{"values": [0, 1]}])

        assert {trace.return_value for trace in parsing.traces} == {False, True}
        assert {trace.return_value for trace in collections.traces} == {False, True}

    def test_text_and_path_models_are_concretely_validated(self) -> None:
        text_trace = ConcolicEngine(text_pack_target).trace({"code": 66})
        path_trace = ConcolicEngine(path_pack_target).trace(
            {"root": "/tmp", "name": "entry"}
        )

        assert text_trace.return_value is True
        assert path_trace.return_value is True
        assert all(branch.predicate is not None for branch in text_trace.branches)
        assert all(branch.predicate is not None for branch in path_trace.branches)
        assert text_trace.is_sound and path_trace.is_sound

    def test_url_sanitization_slice_and_split_models(self) -> None:
        url_result = ConcolicEngine(
            url_string_model_target,
            ConcolicOptions(max_iterations=16, samples_per_frontier=4),
        ).explore([{"text": ""}])
        split_result = ConcolicEngine(
            split_model_target,
            ConcolicOptions(max_iterations=20, samples_per_frontier=4),
        ).explore([{"text": ""}])

        assert {trace.return_value for trace in url_result.traces} == {False, True}
        assert {trace.return_value for trace in split_result.traces} == {False, True}

    def test_literal_backslash_unicode_text_is_not_reinterpreted(self) -> None:
        literal = "//\\u{4000}/"
        expression = exact_string_val(literal)

        assert z3.simplify(z3.Length(expression)).as_long() == len(literal)
        assert z3.simplify(z3.IndexOf(expression, z3.StringVal("/"), 2)).as_long() == 10


class TestPropertyIntegrations:
    def test_pytest_assertion_reports_generated_counterexample(self) -> None:
        def nonnegative(trace):
            assert trace.return_value >= 0, "absolute value must be nonnegative"

        with pytest.raises(ConcolicAssertionError) as raised:
            assert_concolic(
                buggy_absolute,
                [{"value": 1}],
                [nonnegative],
                ConcolicOptions(max_iterations=8),
            )

        assert raised.value.report.violations
        assert raised.value.report.violations[0].inputs["value"] < 0

    def test_pytest_parameters_and_hypothesis_strategy_use_campaign_inputs(
        self,
    ) -> None:
        result = ConcolicEngine(
            arithmetic_target,
            ConcolicOptions(max_iterations=5),
        ).explore([{"x": 1}])

        parameters = pytest_parameters(result)
        strategy = hypothesis_strategy(result)

        assert len(parameters) == len(result.traces)
        assert "sampled_from" in repr(strategy)


class TestRichInputDomains:
    def test_bytes_and_bytearray_are_sampled_and_mutated(self) -> None:
        bytes_result = ConcolicEngine(
            bytes_target,
            ConcolicOptions(max_iterations=16, samples_per_frontier=4),
        ).explore([{"data": b""}])
        bytearray_result = ConcolicEngine(
            bytearray_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        ).explore([{"data": bytearray([0]), "added": 0}])

        assert {trace.return_value for trace in bytes_result.traces} == {False, True}
        assert {trace.return_value for trace in bytearray_result.traces} == {
            False,
            True,
        }
        assert all(
            isinstance(inputs["data"], bytes)
            for inputs in bytes_result.generated_inputs
        )

    def test_sets_support_symbolic_membership(self) -> None:
        result = ConcolicEngine(
            set_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        ).explore([{"values": {1, 2}, "needle": 0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(
            isinstance(inputs["values"], set) for inputs in result.generated_inputs
        )

    def test_ieee_float_inputs_use_qf_fp_sampling(self) -> None:
        result = ConcolicEngine(
            float_target,
            ConcolicOptions(max_iterations=12, samples_per_frontier=4),
        ).explore([{"value": 0.0}])

        assert {trace.return_value for trace in result.traces} == {False, True}
        assert all(
            isinstance(inputs["value"], float) for inputs in result.generated_inputs
        )
        assert all(trace.is_sound for trace in result.traces)

    def test_bounded_shape_expansion_covers_length_classes(self) -> None:
        result = ConcolicEngine(
            shape_target,
            ConcolicOptions(
                max_iterations=16,
                expand_input_shapes=True,
                max_shape_variants=8,
            ),
        ).explore([{"values": [1, 2]}])

        assert {trace.return_value for trace in result.traces} == {
            "empty",
            "small",
            "large",
        }


class TestGapExplanation:
    def test_gap_report_ranks_opaque_blocked_frontier(self) -> None:
        result = ConcolicEngine(
            side_effect_target,
            ConcolicOptions(explain_gaps=True, max_iterations=4),
        ).explore([{"x": 1}])

        assert result.gaps is not None
        assert result.gaps.blocked_frontiers >= 1
        assert result.gaps.hotspots
        assert result.gaps.hotspots[0].input_types["x"] == "int"
        assert result.gaps.hotspots[0].recommendation

    def test_explain_gaps_cli_prints_and_writes_json(
        self, tmp_path: Path, capsys
    ) -> None:
        destination = tmp_path / "gaps.json"

        status = concolic_main(
            [
                f"{__name__}:side_effect_target",
                "--seed",
                '{"x": 1}',
                "--direct",
                "--explain-gaps",
                "--gaps-json",
                str(destination),
                "--max-iterations",
                "3",
            ]
        )

        captured = capsys.readouterr()
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert status == 0
        assert "Blocked frontiers:" in captured.err
        assert payload["hotspots"]


class TestArtifactsAndShrinking:
    def test_solver_guided_shrinking_revalidates_failure(self) -> None:
        engine = ConcolicEngine(shrink_target)
        original = engine.trace({"x": 100})

        result = engine.shrink(original)

        assert result.changed
        assert result.shrunk.inputs == {"x": 10}
        assert result.shrunk.outcome == ExecutionOutcome.RAISED
        assert result.shrunk.exception_type == "builtins.RuntimeError"

    def test_artifact_round_trip_and_replay(self, tmp_path: Path) -> None:
        engine = ConcolicEngine(arithmetic_target)
        result = engine.explore([{"x": 1}])
        artifact = tmp_path / "campaign.json"
        store = ArtifactStore()

        store.write(artifact, "test:arithmetic_target", result, {"suite": "unit"})
        payload = store.read(artifact)
        replay = store.replay(artifact, engine)

        assert payload["schema"] == "aria.concolic/v1"
        assert payload["metadata"] == {"suite": "unit"}
        assert replay.stable
        assert len(replay.traces) == len(result.traces)

    def test_pytest_regression_generation(self, tmp_path: Path) -> None:
        engine = ConcolicEngine(shrink_target)
        failure = engine.trace({"x": 10})
        destination = tmp_path / "test_generated.py"

        ArtifactStore().write_pytest_regressions(
            destination,
            __name__,
            "shrink_target",
            [failure],
        )

        generated = destination.read_text(encoding="utf-8")
        assert "test_concolic_regression_0" in generated
        assert "builtins" in generated
        namespace = {}
        exec(compile(generated, str(destination), "exec"), namespace)
        namespace["test_concolic_regression_0"]()

    def test_generated_regression_supports_positional_only_target(
        self, tmp_path: Path
    ) -> None:
        engine = ConcolicEngine(positional_failure_target)
        failure = engine.trace({"value": 11, "threshold": 10})
        destination = tmp_path / "test_positional_generated.py"

        ArtifactStore().write_pytest_regressions(
            destination,
            __name__,
            "positional_failure_target",
            [failure],
        )

        namespace = {}
        generated = destination.read_text(encoding="utf-8")
        exec(compile(generated, str(destination), "exec"), namespace)
        namespace["test_concolic_regression_0"]()

    def test_async_replay_and_shrinking(self, tmp_path: Path) -> None:
        async def run() -> None:
            engine = ConcolicEngine(async_failure_target)
            failure = await engine.atrace({"x": 100})
            result = ConcolicResult(
                traces=[failure],
                generated_inputs=[{"x": 100}],
                exhausted=True,
                iterations=1,
                pending_frontiers=0,
            )
            artifact = tmp_path / "async-campaign.json"
            store = ArtifactStore()
            store.write(artifact, f"{__name__}:async_failure_target", result)

            replay = await store.areplay(artifact, engine)
            shrunk = await engine.ashrink(failure)

            assert replay.stable
            assert shrunk.changed
            assert shrunk.shrunk.inputs == {"x": 10}

        asyncio.run(run())

    def test_structured_failure_shrinking_reconstructs_container(self) -> None:
        engine = ConcolicEngine(structured_failure_target)
        failure = engine.trace({"values": [100, 7]})

        result = engine.shrink(failure)

        assert result.changed
        assert result.shrunk.inputs["values"][0] == 10
        assert isinstance(result.shrunk.inputs["values"], list)

    def test_structured_artifact_replay_preserves_container_types(
        self, tmp_path: Path
    ) -> None:
        engine = ConcolicEngine(structured_target)
        inputs = {"values": [1, 2], "config": {"limit": 20}, "pair": (3, 4)}
        trace = engine.trace(inputs)
        result = ConcolicResult(
            traces=[trace],
            generated_inputs=[inputs],
            exhausted=True,
            iterations=1,
            pending_frontiers=0,
        )
        artifact = tmp_path / "structured-campaign.json"
        store = ArtifactStore()
        store.write(artifact, f"{__name__}:structured_target", result)

        replay = store.replay(artifact, engine)

        assert replay.stable
        assert isinstance(replay.traces[0].inputs["values"], list)
        assert isinstance(replay.traces[0].inputs["pair"], tuple)

    def test_cli_runs_campaign_and_writes_artifact(
        self, tmp_path: Path, capsys
    ) -> None:
        artifact = tmp_path / "cli-campaign.json"

        status = concolic_main(
            [
                f"{__name__}:arithmetic_target",
                "--seed",
                '{"x": 1}',
                "--direct",
                "--artifact",
                str(artifact),
                "--max-iterations",
                "4",
            ]
        )

        captured = capsys.readouterr()
        assert status == 0
        assert artifact.exists()
        assert '"iterations"' in captured.out


def test_async_target_has_task_local_runtime_state() -> None:
    async def run() -> None:
        engine = ConcolicEngine(async_target)
        low, high = await asyncio.gather(
            engine.atrace({"x": 1}), engine.atrace({"x": 9})
        )

        assert low.return_value == 0
        assert high.return_value == 1
        assert low.branches[0].taken is False
        assert high.branches[0].taken is True
        assert low.is_sound and high.is_sound

    asyncio.run(run())
