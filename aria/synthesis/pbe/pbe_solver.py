"""Programming by Example solver using typed version-space filtering."""

import time
from typing import Any, Callable, Dict, List, Optional

from .expression_generators import generate_expressions_for_task, rank_expressions
from .expressions import Expression, Theory
from .task import PBETask
from .vsa import VSAlgebra, VersionSpace

Oracle = Callable[[Dict[str, Any]], Any]


class SynthesisResult:
    """Result of program synthesis."""

    def __init__(
        self,
        success: bool,
        expression: Optional[Expression] = None,
        version_space: Optional[VersionSpace] = None,
        message: str = "",
        ranked_expressions: Optional[List[Expression]] = None,
        distinguishing_inputs: Optional[List[Dict[str, Any]]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        task: Optional[PBETask] = None,
    ):
        self.success = success
        self.expression = expression
        self.version_space = version_space
        self.message = message
        self.ranked_expressions = ranked_expressions or []
        self.distinguishing_inputs = distinguishing_inputs or []
        self.statistics = statistics or {}
        self.task = task

    @property
    def is_ambiguous(self) -> bool:
        """Return whether synthesis left multiple consistent candidates."""
        return (
            self.success
            and self.version_space is not None
            and len(self.version_space) > 1
        )

    def __str__(self) -> str:
        if self.is_ambiguous and self.expression and self.version_space:
            return (
                "Synthesis ambiguous: selected "
                f"{self.expression} from {len(self.version_space)} programs"
            )
        if self.success and self.expression:
            return f"Synthesis successful: {self.expression}"
        if self.success and self.version_space:
            return f"Synthesis found {len(self.version_space)} possible programs"
        return f"Synthesis failed: {self.message}"


class PBESolver:
    """Programming by Example solver with typed ranking and validation."""

    def __init__(
        self,
        max_expression_depth: int = 3,
        timeout: float = 30.0,
        max_counterexamples: int = 10,
        theory_hint: Optional[Theory] = None,
        max_candidates: int = 2000,
        bitwidth: int = 32,
    ):
        self.max_expression_depth = max_expression_depth
        self.timeout = timeout
        self.max_counterexamples = max_counterexamples
        self.theory_hint = theory_hint
        self.max_candidates = max_candidates
        self.bitwidth = bitwidth

    def synthesize(
        self, examples: List[Dict[str, Any]], theory: Optional[Theory] = None
    ) -> SynthesisResult:
        """Synthesize a program from input-output examples."""
        try:
            task = self._resolve_task(examples, theory=theory)
        except ValueError as error:
            return SynthesisResult(False, message=str(error))
        return self._synthesize_task(task)

    def synthesize_with_oracle(
        self,
        examples: List[Dict[str, Any]],
        oracle: Oracle,
        theory: Optional[Theory] = None,
        max_refinement_rounds: Optional[int] = None,
    ) -> SynthesisResult:
        """Refine an ambiguous synthesis result by querying an external oracle."""
        try:
            task = self._resolve_task(examples, theory=theory)
        except ValueError as error:
            return SynthesisResult(False, message=str(error))

        deadline = time.time() + self.timeout
        refinement_limit = (
            max_refinement_rounds
            if max_refinement_rounds is not None
            else self.max_counterexamples
        )
        refinement_limit = max(0, refinement_limit)

        result = self._synthesize_task(task, deadline=deadline)
        if not result.success:
            return self._apply_refinement_statistics(
                result,
                initial_consistent=0,
                refinement_rounds=0,
                oracle_calls=0,
                ambiguity_resolved=False,
            )

        initial_consistent = self._consistent_expression_count(result)
        refinement_rounds = 0
        oracle_calls = 0
        stop_reason = ""

        while result.is_ambiguous:
            if refinement_rounds >= refinement_limit:
                stop_reason = "Refinement stopped after reaching the round limit"
                break

            if self._remaining_time(deadline) <= 0:
                stop_reason = "Refinement stopped because the timeout budget expired"
                break

            distinguishing_input = self._find_refinement_counterexample(result)
            if distinguishing_input is None:
                stop_reason = "Refinement stopped because no distinguishing input exists"
                break

            oracle_calls += 1
            try:
                oracle_output = oracle(dict(distinguishing_input))
            except Exception as error:
                failure = self._inherit_failure(
                    result,
                    f"Oracle failed during refinement round {refinement_rounds + 1}: "
                    f"{error}",
                )
                return self._apply_refinement_statistics(
                    failure,
                    initial_consistent=initial_consistent,
                    refinement_rounds=refinement_rounds,
                    oracle_calls=oracle_calls,
                    ambiguity_resolved=False,
                )

            if result.task is None or not result.task.output_matches(oracle_output):
                failure = self._inherit_failure(
                    result,
                    "Oracle returned a value incompatible with the task output type",
                )
                return self._apply_refinement_statistics(
                    failure,
                    initial_consistent=initial_consistent,
                    refinement_rounds=refinement_rounds,
                    oracle_calls=oracle_calls,
                    ambiguity_resolved=False,
                )

            refinement_rounds += 1
            if self._remaining_time(deadline) <= 0:
                stop_reason = "Refinement stopped because the timeout budget expired"
                break
            next_task = result.task.with_example(
                {**distinguishing_input, "output": oracle_output}
            )
            next_result = self._synthesize_task(next_task, deadline=deadline)
            if not next_result.success:
                if "Timeout" in next_result.message:
                    stop_reason = (
                        "Refinement stopped because the timeout budget expired"
                    )
                    break
                return self._apply_refinement_statistics(
                    next_result,
                    initial_consistent=initial_consistent,
                    refinement_rounds=refinement_rounds,
                    oracle_calls=oracle_calls,
                    ambiguity_resolved=False,
                )
            result = next_result

        ambiguity_resolved = initial_consistent > 1 and not result.is_ambiguous
        return self._apply_refinement_statistics(
            result,
            initial_consistent=initial_consistent,
            refinement_rounds=refinement_rounds,
            oracle_calls=oracle_calls,
            ambiguity_resolved=ambiguity_resolved,
            note=stop_reason,
        )

    def verify(self, expression: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Verify that an expression is consistent with all examples."""
        task = PBETask.from_examples(
            examples,
            theory_hint=expression.theory,
            bitwidth=self.bitwidth,
        )
        algebra = self._create_algebra(task)
        return algebra.is_consistent(expression, task.as_examples())

    def generate_counterexample(
        self,
        expressions: List[Expression],
        examples: List[Dict[str, Any]],
        task: Optional[PBETask] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate an input that distinguishes a set of expressions."""
        if not expressions:
            return None

        theory = expressions[0].theory

        def expression_generator() -> List[Expression]:
            return list(expressions)

        if task is None:
            try:
                task = PBETask.from_examples(
                    examples,
                    theory_hint=theory,
                    bitwidth=self.bitwidth,
                )
            except ValueError:
                task = None

        observed_examples = task.as_examples() if task is not None else examples
        algebra = VSAlgebra(
            theory,
            expression_generator,
            input_types=task.input_types if task is not None else None,
            observed_examples=observed_examples,
            bitwidth=self.bitwidth,
        )
        return algebra.find_counterexample(
            VersionSpace(set(expressions)),
            observed_examples,
        )

    def minimize_version_space(self, version_space: VersionSpace) -> VersionSpace:
        """Minimize a version space by removing redundant expressions."""
        theory = version_space.theory
        if theory is None:
            return version_space

        algebra = VSAlgebra(theory)
        return algebra.minimize(version_space)

    def sample_from_version_space(
        self, version_space: VersionSpace, n: int = 1
    ) -> List[Expression]:
        """Sample expressions from a version space."""
        theory = version_space.theory
        if theory is None:
            return []

        algebra = VSAlgebra(theory)
        return algebra.sample(version_space, n)

    def _resolve_task(
        self, examples: List[Dict[str, Any]], theory: Optional[Theory] = None
    ) -> PBETask:
        return PBETask.from_examples(
            examples,
            theory_hint=theory or self.theory_hint,
            bitwidth=self.bitwidth,
        )

    def _create_algebra(self, task: PBETask) -> VSAlgebra:
        return VSAlgebra(
            task.theory,
            self._expression_generator(task),
            enable_caching=True,
            max_workers=4,
            input_types=task.input_types,
            observed_examples=task.as_examples(),
            bitwidth=self.bitwidth,
        )

    def _expression_generator(self, task: PBETask) -> Callable[[], List[Expression]]:
        def expression_generator() -> List[Expression]:
            return generate_expressions_for_task(
                task,
                max_depth=self.max_expression_depth,
                max_candidates=self.max_candidates,
            )

        return expression_generator

    def _synthesize_task(
        self, task: PBETask, deadline: Optional[float] = None
    ) -> SynthesisResult:
        task_examples = task.as_examples()
        remaining = self._remaining_time(deadline)
        if deadline is not None and remaining <= 0:
            return SynthesisResult(False, message="Timeout during enumeration", task=task)

        start_time = time.time()
        try:
            expressions = generate_expressions_for_task(
                task,
                max_depth=self.max_expression_depth,
                max_candidates=self.max_candidates,
            )
        except Exception as error:
            return SynthesisResult(
                False,
                message=f"Failed to generate expressions: {error}",
                task=task,
            )

        if not expressions:
            return SynthesisResult(False, message="No expressions generated", task=task)

        if deadline is not None and self._remaining_time(deadline) <= 0:
            return SynthesisResult(False, message="Timeout during enumeration", task=task)

        algebra = self._create_algebra(task)
        filtered_vs = algebra.filter_consistent(VersionSpace(set(expressions)), task_examples)
        if filtered_vs.is_empty():
            return SynthesisResult(
                False,
                message="No consistent programs found",
                task=task,
            )

        ranked = rank_expressions(filtered_vs.expressions)
        best = ranked[0]
        statistics = task.statistics()
        statistics.update(
            {
                "generated_expressions": len(expressions),
                "consistent_expressions": len(filtered_vs),
                "search_time_seconds": round(time.time() - start_time, 6),
                "ambiguity_count": max(0, len(filtered_vs) - 1),
            }
        )
        statistics.update(algebra.get_cache_stats())

        if len(filtered_vs) == 1:
            expanded_vs = self._expand_ambiguous_version_space(
                task,
                best,
                deadline=deadline,
            )
            if len(expanded_vs) > 1:
                expanded_ranked = rank_expressions(expanded_vs.expressions)
                distinguishing_inputs = self._find_distinguishing_inputs(
                    algebra,
                    expanded_vs,
                    task_examples,
                )
                statistics["consistent_expressions"] = len(expanded_vs)
                statistics["ambiguity_count"] = max(0, len(expanded_vs) - 1)
                return SynthesisResult(
                    True,
                    expression=expanded_ranked[0],
                    version_space=expanded_vs,
                    message=(
                        "Ambiguous specification: multiple programs remain "
                        "consistent after semantic expansion"
                    ),
                    ranked_expressions=expanded_ranked[:10],
                    distinguishing_inputs=distinguishing_inputs,
                    statistics=statistics,
                    task=task,
                )

            return SynthesisResult(
                True,
                expression=best,
                version_space=filtered_vs,
                ranked_expressions=ranked[:10],
                statistics=statistics,
                task=task,
            )

        distinguishing_inputs = self._find_distinguishing_inputs(
            algebra,
            filtered_vs,
            task_examples,
        )
        return SynthesisResult(
            True,
            expression=best,
            version_space=filtered_vs,
            message=(
                f"Ambiguous specification: {len(filtered_vs)} programs satisfy the "
                "examples; returning the simplest candidate"
            ),
            ranked_expressions=ranked[:10],
            distinguishing_inputs=distinguishing_inputs,
            statistics=statistics,
            task=task,
        )

    def _find_refinement_counterexample(
        self, result: SynthesisResult
    ) -> Optional[Dict[str, Any]]:
        if result.distinguishing_inputs:
            return dict(result.distinguishing_inputs[0])

        if result.version_space is None or result.task is None:
            return None

        return self.generate_counterexample(
            list(result.version_space.expressions),
            result.task.as_examples(),
            task=result.task,
        )

    def _expand_ambiguous_version_space(
        self,
        task: PBETask,
        best_expression: Expression,
        deadline: Optional[float] = None,
    ) -> VersionSpace:
        if deadline is not None and self._remaining_time(deadline) <= 0:
            return VersionSpace({best_expression})

        expansion_limit = max(self.max_candidates, 200)
        expressions = generate_expressions_for_task(
            task,
            max_depth=self.max_expression_depth,
            max_candidates=expansion_limit,
            deduplicate_observationally=False,
        )
        if deadline is not None and self._remaining_time(deadline) <= 0:
            return VersionSpace({best_expression})

        algebra = VSAlgebra(
            task.theory,
            input_types=task.input_types,
            observed_examples=task.as_examples(),
            bitwidth=self.bitwidth,
        )
        expanded_vs = algebra.filter_consistent(
            VersionSpace(set(expressions)),
            task.as_examples(),
        )
        expanded_vs = algebra.minimize(expanded_vs)
        if expanded_vs.is_empty():
            return VersionSpace({best_expression})
        if best_expression not in expanded_vs.expressions:
            expanded_vs = algebra.minimize(
                expanded_vs.union(VersionSpace({best_expression}))
            )
        return expanded_vs

    def _find_distinguishing_inputs(
        self,
        algebra: VSAlgebra,
        version_space: VersionSpace,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Suggest additional distinguishing inputs without fabricating outputs."""
        suggestions: List[Dict[str, Any]] = []
        seen = set()

        for _ in range(self.max_counterexamples):
            counterexample = algebra.find_counterexample(
                version_space,
                examples + suggestions,
            )
            if counterexample is None:
                break
            frozen = tuple(sorted(counterexample.items()))
            if frozen in seen:
                break
            seen.add(frozen)
            suggestions.append(counterexample)

        return suggestions

    def _apply_refinement_statistics(
        self,
        result: SynthesisResult,
        *,
        initial_consistent: int,
        refinement_rounds: int,
        oracle_calls: int,
        ambiguity_resolved: bool,
        note: str = "",
    ) -> SynthesisResult:
        statistics = dict(result.statistics)
        statistics.update(
            {
                "initial_consistent_expressions": initial_consistent,
                "final_consistent_expressions": self._consistent_expression_count(
                    result
                ),
                "refinement_rounds": refinement_rounds,
                "oracle_calls": oracle_calls,
                "ambiguity_resolved": ambiguity_resolved,
            }
        )
        result.statistics = statistics
        if note:
            if result.message:
                result.message = f"{result.message}. {note}"
            else:
                result.message = note
        return result

    def _consistent_expression_count(self, result: SynthesisResult) -> int:
        if result.version_space is None:
            return 0
        return len(result.version_space)

    def _inherit_failure(self, result: SynthesisResult, message: str) -> SynthesisResult:
        return SynthesisResult(
            False,
            expression=result.expression,
            version_space=result.version_space,
            message=message,
            ranked_expressions=list(result.ranked_expressions),
            distinguishing_inputs=list(result.distinguishing_inputs),
            statistics=dict(result.statistics),
            task=result.task,
        )

    def _remaining_time(self, deadline: Optional[float]) -> float:
        if deadline is None:
            return self.timeout
        return deadline - time.time()
