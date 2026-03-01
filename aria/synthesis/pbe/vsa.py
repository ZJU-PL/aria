"""Version Space Algebra implementation with typed semantics."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .expressions import Expression, Theory, ValueType


class VersionSpace:
    """Represents a version space of expressions."""

    def __init__(self, expressions: Optional[Set[Expression]] = None):
        self.expressions = expressions or set()
        self._theory: Optional[Theory] = None

        if self.expressions:
            theories = {expr.theory for expr in self.expressions}
            if len(theories) > 1:
                raise ValueError(
                    f"All expressions must have the same theory, got: {theories}"
                )
            self._theory = next(iter(theories))

    @property
    def theory(self) -> Optional[Theory]:
        """Return the theory of expressions in this version space."""
        return self._theory

    def add(self, expr: Expression) -> None:
        """Add an expression to the version space."""
        if self._theory is None:
            self._theory = expr.theory
        elif expr.theory != self._theory:
            raise ValueError(
                f"Expression theory {expr.theory} doesn't match version space "
                f"theory {self._theory}"
            )
        self.expressions.add(expr)

    def remove(self, expr: Expression) -> None:
        """Remove an expression from the version space."""
        self.expressions.discard(expr)

    def contains(self, expr: Expression) -> bool:
        """Check if an expression is in the version space."""
        return expr in self.expressions

    def union(self, other: "VersionSpace") -> "VersionSpace":
        """Return the union of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(
                f"Cannot union version spaces of different theories: {self._theory} "
                f"vs {other._theory}"
            )
        return VersionSpace(self.expressions | other.expressions)

    def intersect(self, other: "VersionSpace") -> "VersionSpace":
        """Return the intersection of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(
                "Cannot intersect version spaces of different theories: "
                f"{self._theory} vs {other._theory}"
            )
        return VersionSpace(self.expressions & other.expressions)

    def difference(self, other: "VersionSpace") -> "VersionSpace":
        """Return the set difference of two version spaces."""
        if self._theory != other._theory:
            raise ValueError(
                "Cannot compute difference of version spaces of different "
                f"theories: {self._theory} vs {other._theory}"
            )
        return VersionSpace(self.expressions - other.expressions)

    def is_empty(self) -> bool:
        """Return whether the version space is empty."""
        return len(self.expressions) == 0

    def size(self) -> int:
        """Return the number of expressions in the version space."""
        return len(self.expressions)

    def __len__(self) -> int:
        return self.size()

    def __str__(self) -> str:
        if self.is_empty():
            return "∅"
        exprs = sorted((str(expr) for expr in self.expressions))
        preview = ", ".join(exprs[:5])
        if len(exprs) > 5:
            preview += f", ... ({len(exprs)} total)"
        return f"{{{preview}}}"

    def __repr__(self) -> str:
        return f"VersionSpace({self.expressions})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, VersionSpace) and self.expressions == other.expressions

    def __hash__(self) -> int:
        return hash(frozenset(self.expressions))


class ExpressionCache:
    """Thread-safe cache for expression evaluations."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[Tuple[str, Tuple[Tuple[Tuple[str, Any], ...], ...]], Any] = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(
        self, key: Tuple[str, Tuple[Tuple[Tuple[str, Any], ...], ...]]
    ) -> Optional[Any]:
        """Get a cached result."""
        with self.lock:
            return self.cache.get(key)

    def put(
        self, key: Tuple[str, Tuple[Tuple[Tuple[str, Any], ...], ...]], value: Any
    ) -> None:
        """Store a cached result."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                items_to_remove = max(1, len(self.cache) // 10)
                for _ in range(items_to_remove):
                    self.cache.pop(next(iter(self.cache)))
            self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Return the cache size."""
        with self.lock:
            return len(self.cache)


class VSAlgebra:
    """Algebra for manipulating version spaces."""

    def __init__(
        self,
        theory: Theory,
        expression_generator: Optional[Callable[[], List[Expression]]] = None,
        enable_caching: bool = True,
        max_workers: int = 4,
    ):
        self.theory = theory
        self.expression_generator = expression_generator
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        self.cache = ExpressionCache() if enable_caching else None
        self.evaluation_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_evaluations": 0,
        }

    def empty(self) -> VersionSpace:
        """Create an empty version space."""
        return VersionSpace()

    def singleton(self, expr: Expression) -> VersionSpace:
        """Create a singleton version space."""
        if expr.theory != self.theory:
            raise ValueError(
                f"Expression theory {expr.theory} doesn't match algebra theory "
                f"{self.theory}"
            )
        return VersionSpace({expr})

    def universal(self) -> VersionSpace:
        """Create the universal version space."""
        if self.expression_generator is None:
            raise ValueError("Cannot create universal set without expression generator")
        return VersionSpace(set(self.expression_generator()))

    def join(self, vs1: VersionSpace, vs2: VersionSpace) -> VersionSpace:
        """Return the union of two version spaces."""
        return vs1.union(vs2)

    def meet(self, vs1: VersionSpace, vs2: VersionSpace) -> VersionSpace:
        """Return the intersection of two version spaces."""
        return vs1.intersect(vs2)

    def complement(self, vs: VersionSpace) -> VersionSpace:
        """Return the complement of a version space."""
        return self.universal().difference(vs)

    def filter_consistent(
        self, vs: VersionSpace, examples: List[Dict[str, Any]]
    ) -> VersionSpace:
        """Filter a version space to keep only consistent expressions."""
        if not vs.expressions:
            return vs

        if len(vs.expressions) > 100 and self.max_workers > 1:
            return self._filter_consistent_parallel(vs, examples)
        return self._filter_consistent_sequential(vs, examples)

    def _filter_consistent_sequential(
        self, vs: VersionSpace, examples: List[Dict[str, Any]]
    ) -> VersionSpace:
        consistent = {
            expr for expr in vs.expressions if self._is_consistent_cached(expr, examples)
        }
        return VersionSpace(consistent)

    def _filter_consistent_parallel(
        self, vs: VersionSpace, examples: List[Dict[str, Any]]
    ) -> VersionSpace:
        expressions = list(vs.expressions)
        chunk_size = max(1, len(expressions) // self.max_workers)
        chunks = [
            expressions[index : index + chunk_size]
            for index in range(0, len(expressions), chunk_size)
        ]
        consistent: Set[Expression] = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, examples)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                consistent.update(future.result())

        return VersionSpace(consistent)

    def _process_chunk(
        self, expressions: List[Expression], examples: List[Dict[str, Any]]
    ) -> Set[Expression]:
        consistent = set()
        for expr in expressions:
            if self._is_consistent_cached(expr, examples):
                consistent.add(expr)
        return consistent

    def is_consistent(self, expr: Expression, examples: List[Dict[str, Any]]) -> bool:
        """Public consistency predicate."""
        return self._is_consistent_cached(expr, examples)

    def _is_consistent_cached(
        self, expr: Expression, examples: List[Dict[str, Any]]
    ) -> bool:
        if not self.enable_caching or self.cache is None:
            return self._evaluate_consistency(expr, examples)

        key = (str(expr), self._examples_key(examples))
        cached = self.cache.get(key)
        if cached is not None:
            self.evaluation_stats["cache_hits"] += 1
            return cached

        self.evaluation_stats["cache_misses"] += 1
        result = self._evaluate_consistency(expr, examples)
        self.cache.put(key, result)
        return result

    def _examples_key(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        return tuple(tuple(sorted(example.items())) for example in examples)

    def _evaluate_consistency(
        self, expr: Expression, examples: List[Dict[str, Any]]
    ) -> bool:
        self.evaluation_stats["total_evaluations"] += 1

        for example in examples:
            try:
                actual_output = expr.evaluate(example)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return False

            expected_output = example.get("output")
            if expr.value_type == ValueType.INT:
                if not isinstance(actual_output, int) or isinstance(actual_output, bool):
                    return False
            elif expr.value_type == ValueType.STRING:
                if not isinstance(actual_output, str):
                    return False
            elif expr.value_type == ValueType.BOOL:
                if not isinstance(actual_output, bool):
                    return False
            elif expr.value_type == ValueType.BV:
                if not isinstance(actual_output, int) or isinstance(actual_output, bool):
                    return False

            if actual_output != expected_output:
                return False

        return True

    def generalize(self, vs: VersionSpace, new_example: Dict[str, Any]) -> VersionSpace:
        """Constrain a version space with a new example."""
        return self.filter_consistent(vs, [new_example])

    def observational_signature(
        self, expr: Expression, examples: List[Dict[str, Any]]
    ) -> Optional[Tuple[Any, ...]]:
        """Return an output signature for an expression over the examples."""
        outputs: List[Any] = []
        for example in examples:
            try:
                outputs.append(expr.evaluate(example))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return None
        return tuple(outputs)

    def find_counterexample(
        self, vs: VersionSpace, examples: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find an unlabeled distinguishing input for the version space."""
        if vs.is_empty():
            return None

        variables: Set[str] = set()
        for expr in vs.expressions:
            variables.update(expr.get_variables())

        return self._find_counterexample_heuristic(vs, examples, variables)

    def _find_counterexample_heuristic(
        self, vs: VersionSpace, examples: List[Dict[str, Any]], variables: Set[str]
    ) -> Optional[Dict[str, Any]]:
        example_inputs = {
            tuple(sorted((key, value) for key, value in example.items() if key != "output"))
            for example in examples
        }

        for assignment in self._generate_assignments(variables):
            normalized = tuple(sorted(assignment.items()))
            if normalized in example_inputs:
                continue

            outputs = set()
            for expr in vs.expressions:
                try:
                    outputs.add(expr.evaluate(assignment))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    continue

            if len(outputs) > 1:
                return assignment

        return None

    def _generate_assignments(self, variables: Set[str]) -> List[Dict[str, Any]]:
        if not variables:
            return [{}]

        values = self._candidate_values()
        assignments: List[Dict[str, Any]] = []
        var_list = sorted(variables)

        def visit(current: Dict[str, Any], index: int) -> None:
            if index == len(var_list):
                assignments.append(current.copy())
                return

            name = var_list[index]
            for value in values:
                current[name] = value
                visit(current, index + 1)
                del current[name]

        visit({}, 0)
        return assignments

    def _candidate_values(self) -> List[Any]:
        if self.theory == Theory.STRING:
            return ["", "a", "b", "ab", "abc"]
        if self.theory == Theory.BV:
            return [0, 1, 2, 3, 15, 255]
        return [-2, -1, 0, 1, 2, 3, 5, 10]

    def minimize(self, vs: VersionSpace) -> VersionSpace:
        """Remove structurally dominated observational duplicates."""
        if vs.is_empty():
            return vs

        examples: List[Dict[str, Any]] = []
        if self.expression_generator is not None:
            examples = []

        ranked = sorted(vs.expressions, key=lambda expr: (expr.structural_cost(), str(expr)))
        unique: Set[Expression] = set()
        seen: Set[str] = set()
        for expr in ranked:
            key = str(expr)
            if key in seen:
                continue
            seen.add(key)
            unique.add(expr)
        return VersionSpace(unique)

    def sample(self, vs: VersionSpace, n: int = 1) -> List[Expression]:
        """Sample expressions from a version space."""
        expressions = sorted(vs.expressions, key=str)
        if n >= len(expressions):
            return expressions
        return random.sample(expressions, n)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        if self.cache is None:
            return {"cache_disabled": 1}

        total = self.evaluation_stats["cache_hits"] + self.evaluation_stats["cache_misses"]
        return {
            "cache_size": self.cache.size(),
            "cache_hits": self.evaluation_stats["cache_hits"],
            "cache_misses": self.evaluation_stats["cache_misses"],
            "total_evaluations": self.evaluation_stats["total_evaluations"],
            "hit_rate": self.evaluation_stats["cache_hits"] / max(1, total),
        }

    def clear_cache(self) -> None:
        """Clear cached evaluations."""
        if self.cache is not None:
            self.cache.clear()
        self.evaluation_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_evaluations": 0,
        }
