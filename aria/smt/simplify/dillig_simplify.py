"""
Implementation of the Dillig algorithm for online constraint simplification.

Based on: "Small Formulas for Large Programs: On-line Constraint Simplification
in Scalable Static Analysis" by Isil Dillig, Thomas Dillig, Alex Aiken (SAS 2010).

FIXME: by LLM. Very likely buggy (to debug..)
"""

import logging
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from z3 import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOLVER_TIMEOUT = 1000  # milliseconds
MAX_ITERATIONS = 100  # prevent infinite loops
CACHE_SIZE = 10000  # maximum cache entries


class SimplificationError(Exception):
    """Base class for simplification errors."""

    pass


class SimplificationTimeoutError(SimplificationError):
    """Raised when solver timeout occurs."""

    pass


class InvalidInputError(SimplificationError):
    """Raised when input expression is invalid."""

    pass


@dataclass
class SimplificationStats:
    """Statistics for simplification process."""

    solver_calls: int = 0
    cache_hits: int = 0
    iterations: int = 0
    time_taken: float = 0.0


class SimplificationCache:
    """LRU cache for solver results."""

    def __init__(self, max_size: int = CACHE_SIZE):
        self.cache: Dict[str, bool] = {}
        self.max_size = max_size

    def get(self, expr: z3.ExprRef) -> Optional[bool]:
        key = expr.sexpr()
        return self.cache.get(key)

    def put(self, expr: z3.ExprRef, result: bool):
        if len(self.cache) >= self.max_size:
            # Simple LRU: just clear half the cache
            keys = list(self.cache.keys())
            for k in keys[: len(keys) // 2]:
                del self.cache[k]
        self.cache[expr.sexpr()] = result


def basic_simplify(expr: ExprRef) -> ExprRef:
    """Apply basic simplification patterns before main algorithm.

    Args:
        expr: Z3 expression to simplify

    Returns:
        Simplified expression

    Raises:
        InvalidInputError: If expression is invalid
    """
    if not is_bool(expr):
        raise InvalidInputError("Expression must be boolean")

    if is_not(expr):
        child = expr.children()[0]
        if is_not(child):  # Double negation
            return child.children()[0]
        if is_true(child):  # Not(True) -> False
            return BoolVal(False, expr.ctx)
        if is_false(child):  # Not(False) -> True
            return BoolVal(True, expr.ctx)

    return expr


def check_satisfiability(
    solver: Solver, expr: ExprRef, cache: SimplificationCache
) -> Optional[bool]:
    """Check if expression is satisfiable with timeout and caching."""
    try:
        # Check cache first
        cached = cache.get(expr)
        if cached is not None:
            return cached

        solver.set("timeout", SOLVER_TIMEOUT)
        result = solver.check()

        if result == sat:
            cache.put(expr, True)
            return True
        elif result == unsat:
            cache.put(expr, False)
            return False
        elif result == unknown:
            raise TimeoutError(f"Solver timeout on expression: {expr}")
        return None
    except Z3Exception as e:
        logger.error(f"Z3 error during satisfiability check: {e}")
        return None


def dillig_simplify(
    expr: ExprRef,
    solver: Optional[Solver] = None,
    ctx: Optional[Context] = None,
    stats: Optional[SimplificationStats] = None,
    cache: Optional[SimplificationCache] = None,
) -> ExprRef:
    """Simplify boolean expression using the Dillig algorithm."""
    start_time = time.time()

    # Initialize tracking objects if needed
    if stats is None:
        stats = SimplificationStats()
    if cache is None:
        cache = SimplificationCache()

    logger.debug(f"Simplifying expression: {expr}")

    # Input validation
    if not is_bool(expr):
        raise InvalidInputError("Expression must be boolean")

    # Apply basic simplifications first
    expr = basic_simplify(expr)

    # Initialize solver if needed
    if solver is None:
        ctx = expr.ctx if ctx is None else ctx
        solver = Solver(ctx=ctx)
        solver.add(BoolVal(True, ctx))
        result = dillig_simplify(expr, solver, ctx, stats, cache)
        stats.time_taken = time.time() - start_time
        return result

    # Base case: leaf node
    if not is_and(expr) and not is_or(expr):
        stats.solver_calls += 2

        # Use a fresh solver for each check to avoid stack issues
        check_solver = Solver(ctx=ctx)
        for c in solver.assertions():
            check_solver.add(c)

        # Check if expression is always false
        check_solver.add(expr)
        if check_satisfiability(check_solver, expr, cache) is False:
            return BoolVal(False, ctx)

        # Check if expression is always true
        check_solver = Solver(ctx=ctx)
        for c in solver.assertions():
            check_solver.add(c)
        check_solver.add(Not(expr))
        if check_satisfiability(check_solver, Not(expr), cache) is False:
            return BoolVal(True, ctx)

        return expr

    # Recursive case: AND/OR
    children = list(expr.children())
    unique_children: List[ExprRef] = []
    seen: Set[str] = set()

    # First pass: basic simplification and deduplication
    for child in children:
        child = basic_simplify(child)

        if child.eq(BoolVal(True, ctx)):
            if is_or(expr):
                return BoolVal(True, ctx)
            continue
        elif child.eq(BoolVal(False, ctx)):
            if is_and(expr):
                return BoolVal(False, ctx)
            continue

        child_str = child.sexpr()
        if child_str not in seen:
            seen.add(child_str)
            unique_children.append(child)

    # Fixed point iteration
    iteration = 0
    while iteration < MAX_ITERATIONS:
        stats.iterations += 1
        changed = False

        for i, ci in enumerate(unique_children):
            others = unique_children[:i] + unique_children[i + 1 :]

            # Compute context formula
            # Build the context in the same Z3 context to avoid cast errors
            if is_or(expr):
                context_children = [Not(c) for c in others]
            else:
                context_children = list(others)

            # And() with no args produces a BoolVal(True) in the default
            # global context, which can mismatch the solver context. Make the
            # identity element explicit with the correct ctx.
            context = And(*context_children) if context_children else BoolVal(True, ctx)

            # Use a fresh solver for each child
            child_solver = Solver(ctx=ctx)
            for c in solver.assertions():
                child_solver.add(c)
            child_solver.add(context)

            try:
                new_ci = dillig_simplify(ci, child_solver, ctx, stats, cache)
            except Z3Exception as e:
                logger.error(f"Z3 error during simplification: {e}")
                new_ci = ci

            if not new_ci.eq(ci):
                changed = True
                unique_children[i] = new_ci

            # Short circuit if possible
            if new_ci.eq(BoolVal(True, ctx)) and is_or(expr):
                return BoolVal(True, ctx)
            elif new_ci.eq(BoolVal(False, ctx)) and is_and(expr):
                return BoolVal(False, ctx)

        if not changed:
            break

        iteration += 1
        if iteration == MAX_ITERATIONS:
            logger.warning("Max iterations reached during simplification")

    # Final cleanup
    non_trivial = []
    for c in unique_children:
        if is_and(expr) and c.eq(BoolVal(True, ctx)):
            continue
        if is_or(expr) and c.eq(BoolVal(False, ctx)):
            continue
        non_trivial.append(c)

    if len(non_trivial) == 0:
        return BoolVal(True, ctx) if is_and(expr) else BoolVal(False, ctx)
    if len(non_trivial) == 1:
        return non_trivial[0]

    return And(non_trivial) if is_and(expr) else Or(non_trivial)


class TestDilligSimplify(unittest.TestCase):
    """Test cases for Dillig simplification algorithm."""

    def setUp(self):
        self.ctx = Context()
        self.x = Bool("x", self.ctx)
        self.y = Bool("y", self.ctx)
        self.z = Bool("z", self.ctx)

    def test_basic_simplification(self):
        """Test basic boolean simplifications."""
        # Double negation
        expr = Not(Not(self.x))
        self.assertEqual(dillig_simplify(expr).sexpr(), self.x.sexpr())

        # True/False cases
        self.assertTrue(dillig_simplify(And(self.x, True)).eq(self.x))
        self.assertTrue(dillig_simplify(Or(self.x, True)).eq(BoolVal(True, self.ctx)))
        self.assertTrue(
            dillig_simplify(And(self.x, False)).eq(BoolVal(False, self.ctx))
        )
        self.assertTrue(dillig_simplify(Or(self.x, False)).eq(self.x))


if __name__ == "__main__":
    unittest.main()
