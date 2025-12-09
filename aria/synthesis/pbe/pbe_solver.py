"""Programming by Example (PBE) Solver using Version Space Algebra.

This module provides a PBE solver that synthesizes programs from
input-output examples using version space algebra.
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Set
from .vsa import VSAlgebra, VersionSpace
from .expressions import Expression, Theory, Variable
from .expression_generators import (
    generate_expressions_for_theory,
    get_theory_from_variables
)


class SynthesisResult:
    """Result of program synthesis."""

    def __init__(self, success: bool, expression: Optional[Expression] = None,
                 version_space: Optional[VersionSpace] = None, message: str = ""):
        self.success = success
        self.expression = expression
        self.version_space = version_space
        self.message = message

    def __str__(self) -> str:
        if self.success and self.expression:
            return f"Synthesis successful: {self.expression}"
        elif self.success and self.version_space:
            return f"Synthesis found {len(self.version_space)} possible programs"
        else:
            return f"Synthesis failed: {self.message}"


class PBESolver:
    """Programming by Example solver using Version Space Algebra."""

    def __init__(self, max_expression_depth: int = 3, timeout: float = 30.0,
                 max_counterexamples: int = 10):
        self.max_expression_depth = max_expression_depth
        self.timeout = timeout
        self.max_counterexamples = max_counterexamples

    def synthesize(self, examples: List[Dict[str, Any]]) -> SynthesisResult:
        """Synthesize a program from input-output examples."""
        if not examples:
            return SynthesisResult(False, message="No examples provided")

        # Infer the theory from examples
        try:
            theory = get_theory_from_variables(examples)
        except ValueError as e:
            return SynthesisResult(False, message=str(e))

        # Extract variable names from examples
        variables = self._extract_variables(examples)

        # Generate initial version space
        try:
            expressions = generate_expressions_for_theory(
                theory, variables, max_depth=self.max_expression_depth
            )
        except Exception as e:
            return SynthesisResult(False, message=f"Failed to generate expressions: {e}")

        if not expressions:
            return SynthesisResult(False, message="No expressions generated")

        # Create VSA algebra
        def expression_generator():
            return generate_expressions_for_theory(
                theory, variables, max_depth=self.max_expression_depth
            )

        algebra = VSAlgebra(theory, expression_generator)

        # Create initial version space with all expressions
        initial_vs = VersionSpace(set(expressions))

        # Iteratively refine the version space with examples
        current_vs = initial_vs
        start_time = time.time()

        for i, example in enumerate(examples):
            # Filter version space to be consistent with this example
            current_vs = algebra.filter_consistent(current_vs, [example])

            if current_vs.is_empty():
                return SynthesisResult(
                    False,
                    message=f"No consistent programs found after example {i+1}"
                )

            # Check timeout
            if time.time() - start_time > self.timeout:
                return SynthesisResult(
                    False,
                    message=f"Timeout after {time.time() - start_time:.2f} seconds"
                )

        # Try to find a unique solution by generating counterexamples
        if len(current_vs) > 1:
            counterexample_result = self._find_unique_solution(
                algebra, current_vs, examples, start_time
            )

            if counterexample_result.success:
                return counterexample_result
            else:
                # Return the version space if we can't find a unique solution
                return SynthesisResult(
                    True,
                    version_space=current_vs,
                    message=f"Found {len(current_vs)} possible programs"
                )

        # We have a unique solution
        unique_expr = list(current_vs.expressions)[0]
        return SynthesisResult(True, expression=unique_expr)

    def _extract_variables(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Extract variable names from examples."""
        variables = set()
        for example in examples:
            for key in example.keys():
                if key != 'output':
                    variables.add(key)
        return list(variables)

    def _find_unique_solution(self, algebra: VSAlgebra, version_space: VersionSpace,
                            examples: List[Dict[str, Any]], start_time: float) -> SynthesisResult:
        """Try to find a unique solution by generating counterexamples."""

        for _ in range(self.max_counterexamples):
            # Check timeout
            if time.time() - start_time > self.timeout:
                return SynthesisResult(
                    False,
                    version_space=version_space,
                    message=f"Timeout while searching for unique solution"
                )

            # Find a counterexample
            counterexample = algebra.find_counterexample(version_space, examples)

            if counterexample is None:
                # No more counterexamples found
                break

            # Evaluate the counterexample with expressions in the version space
            outputs = {}
            for expr in version_space.expressions:
                try:
                    output = expr.evaluate(counterexample)
                    outputs[expr] = output
                except:
                    continue

            if len(set(outputs.values())) == 1:
                # All expressions produce the same output, this isn't a good counterexample
                continue

            # Add the counterexample with the expected output
            # For now, we'll use the output from the first expression as the expected output
            first_expr = next(iter(version_space.expressions))
            counterexample['output'] = outputs.get(first_expr)

            # Filter the version space with this counterexample
            new_vs = algebra.filter_consistent(version_space, [counterexample])

            if new_vs.is_empty():
                return SynthesisResult(
                    False,
                    message="Counterexample eliminated all possible programs"
                )

            version_space = new_vs

            # Check if we now have a unique solution
            if len(version_space) == 1:
                unique_expr = list(version_space.expressions)[0]
                return SynthesisResult(True, expression=unique_expr)
            elif len(version_space) == 0:
                return SynthesisResult(
                    False,
                    message="No consistent programs found after counterexample"
                )

        # Return the current version space if we couldn't find a unique solution
        return SynthesisResult(
            True,
            version_space=version_space,
            message=f"Could not find unique solution, found {len(version_space)} possible programs"
        )

    def verify(self, expression: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Verify that an expression is consistent with all examples."""
        algebra = VSAlgebra(expression.theory)
        return algebra._is_consistent(expression, examples)

    def generate_counterexample(self, expressions: List[Expression],
                              examples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate a counterexample that distinguishes between expressions."""
        if not expressions:
            return None

        # Use the first expression's theory
        theory = expressions[0].theory
        variables = set()
        for expr in expressions:
            variables.update(expr.get_variables())

        def expression_generator():
            return generate_expressions_for_theory(theory, list(variables))

        algebra = VSAlgebra(theory, expression_generator)
        vs = VersionSpace(set(expressions))

        return algebra.find_counterexample(vs, examples)

    def minimize_version_space(self, version_space: VersionSpace) -> VersionSpace:
        """Minimize a version space by removing redundant expressions."""
        theory = version_space.theory
        if theory is None:
            return version_space

        def expression_generator():
            variables = set()
            for expr in version_space.expressions:
                variables.update(expr.get_variables())
            return generate_expressions_for_theory(theory, list(variables))

        algebra = VSAlgebra(theory, expression_generator)
        return algebra.minimize(version_space)

    def sample_from_version_space(self, version_space: VersionSpace, n: int = 1) -> List[Expression]:
        """Sample expressions from a version space."""
        theory = version_space.theory
        if theory is None:
            return []

        def expression_generator():
            variables = set()
            for expr in version_space.expressions:
                variables.update(expr.get_variables())
            return generate_expressions_for_theory(theory, list(variables))

        algebra = VSAlgebra(theory, expression_generator)
        return algebra.sample(version_space, n)
