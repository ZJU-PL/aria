"""SMT-enhanced Programming by Example solver.

This module provides an enhanced PBE solver that integrates with Aria's
SMT solvers for improved verification and counterexample generation.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from .vsa import VSAlgebra, VersionSpace
from .expressions import Expression, Theory
from .expression_generators import (
    generate_expressions_for_theory,
    get_theory_from_variables,
)
from .pbe_solver import SynthesisResult
from .smt_verifier import SMTVerifier
from .expression_to_smt import expression_to_smt


class SMTPBESolver:
    """SMT-enhanced Programming by Example solver."""

    def __init__(
        self,
        max_expression_depth: int = 3,
        timeout: float = 30.0,
        max_counterexamples: int = 10,
        use_smt: bool = True,
    ):
        self.max_expression_depth = max_expression_depth
        self.timeout = timeout
        self.max_counterexamples = max_counterexamples
        self.use_smt = use_smt

        # SMT components
        self.smt_verifier = SMTVerifier() if use_smt else None

    def synthesize(self, examples: List[Dict[str, Any]]) -> SynthesisResult:
        """Synthesize a program from input-output examples using SMT."""
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
            return SynthesisResult(
                False, message=f"Failed to generate expressions: {e}"
            )

        if not expressions:
            return SynthesisResult(False, message="No expressions generated")

        # Create VSA algebra with caching enabled
        def expression_generator():
            return generate_expressions_for_theory(
                theory, variables, max_depth=self.max_expression_depth
            )

        algebra = VSAlgebra(
            theory, expression_generator, enable_caching=True, max_workers=4
        )

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
                    False, message=f"No consistent programs found after example {i+1}"
                )

            # Check timeout
            if time.time() - start_time > self.timeout:
                return SynthesisResult(
                    False,
                    message=f"Timeout after {time.time() - start_time:.2f} seconds",
                )

        # Use SMT for enhanced verification if enabled
        if self.use_smt and self.smt_verifier:
            current_vs = self._smt_filter(current_vs, examples)

        # Check if we have a unique solution or multiple possibilities
        if len(current_vs) == 0:
            return SynthesisResult(False, message="No consistent programs found")
        elif len(current_vs) == 1:
            # We have a unique solution
            unique_expr = list(current_vs.expressions)[0]
            return SynthesisResult(True, expression=unique_expr)
        else:
            # Multiple possible programs - try to find counterexamples
            counterexample_result = self._find_unique_solution_smt(
                algebra, current_vs, examples, start_time
            )

            if counterexample_result.success:
                return counterexample_result
            else:
                # Return the version space if we can't find a unique solution
                return SynthesisResult(
                    True,
                    version_space=current_vs,
                    message=f"Found {len(current_vs)} possible programs",
                )

    def _extract_variables(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Extract variable names from examples."""
        variables = set()
        for example in examples:
            for key in example.keys():
                if key != "output":
                    variables.add(key)
        return list(variables)

    def _smt_filter(
        self, vs: VersionSpace, examples: List[Dict[str, Any]]
    ) -> VersionSpace:
        """Use SMT to filter version space more accurately."""
        if not self.smt_verifier:
            return vs

        consistent_expressions = set()

        for expr in vs.expressions:
            if self.smt_verifier.verify_expression(expr, examples):
                consistent_expressions.add(expr)

        return VersionSpace(consistent_expressions)

    def _find_unique_solution_smt(
        self,
        algebra: VSAlgebra,
        version_space: VersionSpace,
        examples: List[Dict[str, Any]],
        start_time: float,
    ) -> SynthesisResult:
        """Find unique solution using SMT-enhanced counterexample generation."""

        for _ in range(self.max_counterexamples):
            # Check timeout
            if time.time() - start_time > self.timeout:
                return SynthesisResult(
                    False,
                    version_space=version_space,
                    message=f"Timeout while searching for unique solution",
                )

            # Use SMT for better counterexample generation
            if self.smt_verifier:
                counterexample = self.smt_verifier.find_counterexample(
                    list(version_space.expressions), examples
                )
            else:
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
            # Use the output from the first expression as the expected output
            first_expr = next(iter(version_space.expressions))
            counterexample["output"] = outputs.get(first_expr)

            # Filter the version space with this counterexample
            new_vs = algebra.filter_consistent(version_space, [counterexample])

            if new_vs.is_empty():
                return SynthesisResult(
                    False, message="Counterexample eliminated all possible programs"
                )

            version_space = new_vs

            # Check if we now have a unique solution
            if len(version_space) == 1:
                unique_expr = list(version_space.expressions)[0]
                return SynthesisResult(True, expression=unique_expr)
            elif len(version_space) == 0:
                return SynthesisResult(
                    False, message="No consistent programs found after counterexample"
                )

        # Return the current version space if we couldn't find a unique solution
        return SynthesisResult(
            True,
            version_space=version_space,
            message=f"Could not find unique solution, found {len(version_space)} possible programs",
        )

    def verify_with_smt(
        self, expression: Expression, examples: List[Dict[str, Any]]
    ) -> bool:
        """Verify an expression using SMT."""
        if not self.smt_verifier:
            # Fall back to regular verification
            algebra = VSAlgebra(expression.theory)
            return algebra._is_consistent(expression, examples)

        return self.smt_verifier.verify_expression(expression, examples)

    def prove_equivalence_with_smt(self, expr1: Expression, expr2: Expression) -> bool:
        """Prove equivalence of two expressions using SMT."""
        if not self.smt_verifier:
            return False

        return self.smt_verifier.prove_equivalence(expr1, expr2)

    def get_smt_formula(self, expression: Expression) -> str:
        """Get SMT-LIB format for an expression."""
        if not self.smt_verifier:
            return str(expression)

        return self.smt_verifier.get_smt_formula(expression)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the VSA algebra."""
        # This would need to be implemented to return stats from the algebra
        return {"smt_integration": "enabled"}

    def enable_smt_integration(self) -> None:
        """Enable SMT integration."""
        self.use_smt = True
        self.smt_verifier = SMTVerifier()

    def disable_smt_integration(self) -> None:
        """Disable SMT integration."""
        self.use_smt = False
        self.smt_verifier = None
