"""Programming by Example solver using typed version-space filtering."""

import time
from typing import Any, Dict, List, Optional

from .expression_generators import (
    generate_expressions_for_theory,
    get_output_type,
    get_theory_from_variables,
    get_variable_names,
    rank_expressions,
    validate_examples,
)
from .expressions import Expression, Theory
from .vsa import VSAlgebra, VersionSpace


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
    ):
        self.success = success
        self.expression = expression
        self.version_space = version_space
        self.message = message
        self.ranked_expressions = ranked_expressions or []
        self.distinguishing_inputs = distinguishing_inputs or []
        self.statistics = statistics or {}

    def __str__(self) -> str:
        if (
            self.success
            and self.expression
            and self.version_space
            and len(self.version_space) > 1
        ):
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
            validate_examples(examples)
            selected_theory = get_theory_from_variables(
                examples, theory_hint=theory or self.theory_hint
            )
            output_type = get_output_type(examples, selected_theory)
            variables = get_variable_names(examples)
        except ValueError as error:
            return SynthesisResult(False, message=str(error))

        start_time = time.time()

        try:
            expressions = generate_expressions_for_theory(
                selected_theory,
                variables,
                max_depth=self.max_expression_depth,
                output_type=output_type,
                examples=examples,
                bitwidth=self.bitwidth,
                max_candidates=self.max_candidates,
            )
        except Exception as error:
            return SynthesisResult(
                False, message=f"Failed to generate expressions: {error}"
            )

        if not expressions:
            return SynthesisResult(False, message="No expressions generated")

        if time.time() - start_time > self.timeout:
            return SynthesisResult(False, message="Timeout during enumeration")

        def expression_generator() -> List[Expression]:
            return generate_expressions_for_theory(
                selected_theory,
                variables,
                max_depth=self.max_expression_depth,
                output_type=output_type,
                examples=examples,
                bitwidth=self.bitwidth,
                max_candidates=self.max_candidates,
            )

        algebra = VSAlgebra(
            selected_theory,
            expression_generator,
            enable_caching=True,
            max_workers=4,
        )

        filtered_vs = algebra.filter_consistent(
            VersionSpace(set(expressions)), examples
        )
        if filtered_vs.is_empty():
            return SynthesisResult(False, message="No consistent programs found")

        ranked = rank_expressions(filtered_vs.expressions)
        best = ranked[0]
        statistics = {
            "theory": selected_theory.value,
            "output_type": output_type.value,
            "generated_expressions": len(expressions),
            "consistent_expressions": len(filtered_vs),
            "search_time_seconds": round(time.time() - start_time, 6),
        }
        statistics.update(algebra.get_cache_stats())

        if len(filtered_vs) == 1:
            return SynthesisResult(
                True,
                expression=best,
                version_space=filtered_vs,
                ranked_expressions=ranked[:10],
                statistics=statistics,
            )

        distinguishing_inputs = self._find_distinguishing_inputs(
            algebra, filtered_vs, examples
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
        )

    def verify(self, expression: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Verify that an expression is consistent with all examples."""
        validate_examples(examples)
        algebra = VSAlgebra(expression.theory)
        return algebra.is_consistent(expression, examples)

    def generate_counterexample(
        self, expressions: List[Expression], examples: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate an input that distinguishes a set of expressions."""
        if not expressions:
            return None

        variables = set()
        for expr in expressions:
            variables.update(expr.get_variables())

        theory = expressions[0].theory

        def expression_generator() -> List[Expression]:
            return list(expressions)

        algebra = VSAlgebra(theory, expression_generator)
        return algebra.find_counterexample(VersionSpace(set(expressions)), examples)

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
                version_space, examples + suggestions
            )
            if counterexample is None:
                break
            frozen = tuple(sorted(counterexample.items()))
            if frozen in seen:
                break
            seen.add(frozen)
            suggestions.append(counterexample)

        return suggestions
