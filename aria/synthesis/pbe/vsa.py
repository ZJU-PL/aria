"""Version Space Algebra implementation.

This module provides the core algebraic operations for manipulating
version spaces in program synthesis with performance optimizations.
"""

import time
import hashlib
import threading
from typing import Set, List, Dict, Any, Callable, Optional, Union, Tuple
from abc import ABC, abstractmethod
from .expressions import Expression, Theory
from concurrent.futures import ThreadPoolExecutor, as_completed


class VersionSpace:
    """Represents a version space - a set of expressions consistent with examples."""

    def __init__(self, expressions: Set[Expression] = None):
        self.expressions = expressions or set()
        self._theory = None

        # Validate that all expressions have the same theory
        if self.expressions:
            theories = {expr.theory for expr in self.expressions}
            if len(theories) > 1:
                raise ValueError(f"All expressions must have the same theory, got: {theories}")
            self._theory = next(iter(theories))

    @property
    def theory(self) -> Optional[Theory]:
        """Get the theory of expressions in this version space."""
        return self._theory

    def add(self, expr: Expression) -> None:
        """Add an expression to the version space."""
        if self._theory is None:
            self._theory = expr.theory
        elif expr.theory != self._theory:
            raise ValueError(f"Expression theory {expr.theory} doesn't match version space theory {self._theory}")

        self.expressions.add(expr)

    def remove(self, expr: Expression) -> None:
        """Remove an expression from the version space."""
        self.expressions.discard(expr)

    def contains(self, expr: Expression) -> bool:
        """Check if an expression is in the version space."""
        return expr in self.expressions

    def union(self, other: 'VersionSpace') -> 'VersionSpace':
        """Union of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(f"Cannot union version spaces of different theories: {self._theory} vs {other._theory}")

        return VersionSpace(self.expressions | other.expressions)

    def intersect(self, other: 'VersionSpace') -> 'VersionSpace':
        """Intersection of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(f"Cannot intersect version spaces of different theories: {self._theory} vs {other._theory}")

        return VersionSpace(self.expressions & other.expressions)

    def difference(self, other: 'VersionSpace') -> 'VersionSpace':
        """Difference of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(f"Cannot compute difference of version spaces of different theories: {self._theory} vs {other._theory}")

        return VersionSpace(self.expressions - other.expressions)

    def is_empty(self) -> bool:
        """Check if the version space is empty."""
        return len(self.expressions) == 0

    def size(self) -> int:
        """Get the number of expressions in the version space."""
        return len(self.expressions)

    def __len__(self) -> int:
        return self.size()

    def __str__(self) -> str:
        if self.is_empty():
            return "∅"

        exprs_str = ", ".join(str(expr) for expr in list(self.expressions)[:5])
        if len(self.expressions) > 5:
            exprs_str += f", ... ({len(self.expressions)} total)"

        return f"{{{exprs_str}}}"

    def __repr__(self) -> str:
        return f"VersionSpace({self.expressions})"

    def __eq__(self, other) -> bool:
        return isinstance(other, VersionSpace) and self.expressions == other.expressions

    def __hash__(self) -> int:
        # Convert to frozenset for hashing
        return hash(frozenset(self.expressions))


class ExpressionCache:
    """Cache for expression evaluation results."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached result for expression evaluation."""
        with self.lock:
            return self.cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """Cache expression evaluation result."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Simple LRU: remove oldest 10%
                items_to_remove = len(self.cache) // 10
                for _ in range(items_to_remove):
                    self.cache.pop(next(iter(self.cache)))

            self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get cache size."""
        with self.lock:
            return len(self.cache)


class VSAlgebra:
    """Algebra for manipulating version spaces with performance optimizations."""

    def __init__(self, theory: Theory, expression_generator: Callable[[], List[Expression]] = None,
                 enable_caching: bool = True, max_workers: int = 4):
        self.theory = theory
        self.expression_generator = expression_generator
        self.enable_caching = enable_caching
        self.max_workers = max_workers

        # Performance optimizations
        self.cache = ExpressionCache() if enable_caching else None
        self.evaluation_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_evaluations': 0
        }

    def empty(self) -> VersionSpace:
        """Create an empty version space."""
        return VersionSpace()

    def singleton(self, expr: Expression) -> VersionSpace:
        """Create a version space containing a single expression."""
        if expr.theory != self.theory:
            raise ValueError(f"Expression theory {expr.theory} doesn't match algebra theory {self.theory}")

        vs = VersionSpace()
        vs.add(expr)
        return vs

    def universal(self) -> VersionSpace:
        """Create the universal version space (all possible expressions)."""
        if self.expression_generator is None:
            raise ValueError("Cannot create universal set without expression generator")

        expressions = self.expression_generator()
        return VersionSpace(set(expressions))

    def join(self, vs1: VersionSpace, vs2: VersionSpace) -> VersionSpace:
        """Join (union) of two version spaces."""
        return vs1.union(vs2)

    def meet(self, vs1: VersionSpace, vs2: VersionSpace) -> VersionSpace:
        """Meet (intersection) of two version spaces."""
        return vs1.intersect(vs2)

    def complement(self, vs: VersionSpace) -> VersionSpace:
        """Complement of a version space."""
        if self.expression_generator is None:
            raise ValueError("Cannot compute complement without expression generator")

        universal = self.universal()
        return universal.difference(vs)

    def filter_consistent(self, vs: VersionSpace, examples: List[Dict[str, Any]]) -> VersionSpace:
        """Filter version space to keep only expressions consistent with examples."""
        if not vs.expressions:
            return vs

        # Use parallel processing for large version spaces
        if len(vs.expressions) > 100 and self.max_workers > 1:
            return self._filter_consistent_parallel(vs, examples)
        else:
            return self._filter_consistent_sequential(vs, examples)

    def _filter_consistent_sequential(self, vs: VersionSpace, examples: List[Dict[str, Any]]) -> VersionSpace:
        """Sequential filtering with caching."""
        consistent_expressions = set()

        for expr in vs.expressions:
            if self._is_consistent_cached(expr, examples):
                consistent_expressions.add(expr)

        return VersionSpace(consistent_expressions)

    def _filter_consistent_parallel(self, vs: VersionSpace, examples: List[Dict[str, Any]]) -> VersionSpace:
        """Parallel filtering for large version spaces."""
        expressions_list = list(vs.expressions)
        consistent_expressions = set()

        # Split expressions into chunks for parallel processing
        chunk_size = max(1, len(expressions_list) // self.max_workers)
        chunks = [expressions_list[i:i + chunk_size] for i in range(0, len(expressions_list), chunk_size)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for parallel processing
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, examples): chunk
                for chunk in chunks
            }

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                consistent_expressions.update(chunk_results)

        return VersionSpace(consistent_expressions)

    def _process_chunk(self, expressions: List[Expression], examples: List[Dict[str, Any]]) -> Set[Expression]:
        """Process a chunk of expressions for consistency checking."""
        consistent = set()
        for expr in expressions:
            if self._is_consistent_cached(expr, examples):
                consistent.add(expr)
        return consistent

    def _is_consistent(self, expr: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Check if an expression is consistent with all examples."""
        return self._is_consistent_cached(expr, examples)

    def _is_consistent_cached(self, expr: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Check consistency with caching for performance."""
        if not self.enable_caching or self.cache is None:
            return self._evaluate_consistency(expr, examples)

        # Create cache key from expression and examples
        examples_key = str(sorted(examples)) if examples else "no_examples"
        cache_key = f"{str(expr)}:{examples_key}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.evaluation_stats['cache_hits'] += 1
            return cached_result

        # Evaluate and cache result
        self.evaluation_stats['cache_misses'] += 1
        result = self._evaluate_consistency(expr, examples)
        self.cache.put(cache_key, result)

        return result

    def _evaluate_consistency(self, expr: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Evaluate consistency without caching."""
        self.evaluation_stats['total_evaluations'] += 1

        for example in examples:
            try:
                actual_output = expr.evaluate(example)
                expected_output = example.get('output')

                # Handle different theories appropriately
                if expr.theory == Theory.LIA:
                    if not isinstance(actual_output, (int, bool)) or not isinstance(expected_output, (int, bool)):
                        return False
                elif expr.theory == Theory.STRING:
                    if not isinstance(actual_output, str) or not isinstance(expected_output, str):
                        return False

                if actual_output != expected_output:
                    return False

            except (KeyError, TypeError, ZeroDivisionError):
                # Expression evaluation failed or missing variables
                return False

        return True

    def generalize(self, vs: VersionSpace, new_example: Dict[str, Any]) -> VersionSpace:
        """Generalize version space to be consistent with a new example."""
        return self.filter_consistent(vs, [new_example])

    def find_counterexample(self, vs: VersionSpace, examples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a counterexample that distinguishes expressions in the version space."""
        if vs.is_empty():
            return None

        # Get all variables used in the version space
        all_variables = set()
        for expr in vs.expressions:
            all_variables.update(expr.get_variables())

        # Try to find an input that produces different outputs for different expressions
        if self.expression_generator is None:
            # Simple heuristic: try small integer values for variables
            return self._find_counterexample_heuristic(vs, examples, all_variables)

        # More sophisticated counterexample generation would go here
        return self._find_counterexample_heuristic(vs, examples, all_variables)

    def _find_counterexample_heuristic(self, vs: VersionSpace, examples: List[Dict[str, Any]],
                                     variables: Set[str]) -> Optional[Dict[str, Any]]:
        """Heuristic counterexample generation using small integer values."""
        # Simple heuristic: try different small integer assignments
        test_values = [-2, -1, 0, 1, 2, 3, 5, 10]

        for assignment in self._generate_assignments(variables, test_values):
            # Skip assignments that are already in examples
            if assignment in examples:
                continue

            # Check if this assignment would produce different outputs for different expressions
            outputs = set()
            for expr in vs.expressions:
                try:
                    output = expr.evaluate(assignment)
                    outputs.add(output)
                except:
                    continue

            # If we get different outputs, this is a good counterexample
            if len(outputs) > 1:
                assignment['output'] = None  # We don't know the expected output yet
                return assignment

        return None

    def _generate_assignments(self, variables: Set[str], values: List[Any]) -> List[Dict[str, Any]]:
        """Generate all possible assignments of values to variables."""
        if not variables:
            return [{}]

        assignments = []
        var_list = list(variables)

        def generate(current: Dict[str, Any], index: int):
            if index == len(var_list):
                assignments.append(current.copy())
                return

            var = var_list[index]
            for value in values:
                current[var] = value
                generate(current, index + 1)
                del current[var]

        generate({}, 0)
        return assignments

    def minimize(self, vs: VersionSpace) -> VersionSpace:
        """Minimize the version space by removing redundant expressions."""
        if vs.is_empty():
            return vs

        # For now, just return a copy (more sophisticated minimization could be added)
        return VersionSpace(vs.expressions.copy())

    def sample(self, vs: VersionSpace, n: int = 1) -> List[Expression]:
        """Sample expressions from the version space."""
        expressions = list(vs.expressions)
        if n >= len(expressions):
            return expressions

        import random
        return random.sample(expressions, n)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        if self.cache is None:
            return {'cache_disabled': 1}

        return {
            'cache_size': self.cache.size(),
            'cache_hits': self.evaluation_stats['cache_hits'],
            'cache_misses': self.evaluation_stats['cache_misses'],
            'total_evaluations': self.evaluation_stats['total_evaluations'],
            'hit_rate': (self.evaluation_stats['cache_hits'] /
                        max(1, self.evaluation_stats['cache_hits'] + self.evaluation_stats['cache_misses']))
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        if self.cache is not None:
            self.cache.clear()
        self.evaluation_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_evaluations': 0
        }
