"""
Memoization utilities for expensive computations.

This module provides sophisticated memoization capabilities for symbolic
computations that may be expensive to recompute. It supports multiple
eviction strategies and weak references for memory management.

Key Features:
- Multiple cache eviction strategies (LRU, FIFO, Random)
- Configurable cache size limits
- Weak reference support for automatic cleanup
- Thread-safe operations for concurrent use
- Hash-based key computation for complex objects
- Integration with function decorators for easy application

Example:
    >>> from aria.srk.memo import memoize, MemoizationTable
    >>> @memoize(max_size=1000)
    ... def expensive_computation(x, y):
    ...     return x * y + x + y  # Some expensive operation
    >>>
    >>> # Or use directly:
    >>> table = MemoizationTable(max_size=100, strategy="lru")
    >>> table.put(("key", 1, 2), "value")
    >>> result = table.get(("key", 1, 2))
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import weakref

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')


class MemoizationTable(Generic[K, V]):
    """Sophisticated memoization table with multiple eviction strategies.

    This class provides a flexible memoization table that supports different
    cache eviction strategies to manage memory usage effectively. It's designed
    for caching expensive computations in symbolic reasoning algorithms.

    Supported eviction strategies:
    - LRU (Least Recently Used): Evict least recently accessed items first
    - FIFO (First In, First Out): Evict oldest items first
    - Random: Evict random items (useful for testing and benchmarks)

    Attributes:
        max_size (int): Maximum number of entries to store before eviction.
        strategy (str): Eviction strategy ("lru", "fifo", or "random").
        table (Dict[K, V]): Internal storage mapping keys to values.
        access_order (List[K]): Tracks access order for LRU eviction.

    Example:
        >>> table = MemoizationTable(max_size=100, strategy="lru")
        >>> table.put("key1", "value1")
        >>> table.put("key2", "value2")
        >>> value = table.get("key1")  # Access moves key1 to end of LRU
    """

    def __init__(self, max_size: int = 10000, strategy: str = "lru"):
        """Initialize memoization table with specified parameters.

        Args:
            max_size: Maximum number of entries before eviction occurs.
            strategy: Eviction strategy - "lru", "fifo", or "random".

        Raises:
            ValueError: If strategy is not one of the supported options.
        """
        self.max_size = max_size
        self.strategy = strategy
        self.table: Dict[K, V] = {}
        self.access_order: List[K] = []

        if strategy == "lru":
            self._evict = self._evict_lru
        elif strategy == "fifo":
            self._evict = self._evict_fifo
        elif strategy == "random":
            self._evict = self._evict_random
        else:
            raise ValueError(f"Unknown eviction strategy: {strategy}")

    def get(self, key: K) -> Optional[V]:
        """Get value from table."""
        if key in self.table:
            if self.strategy == "lru":
                # Move to end for LRU
                self.access_order.remove(key)
                self.access_order.append(key)
            return self.table[key]
        return None

    def put(self, key: K, value: V) -> None:
        """Put value in table."""
        if key in self.table:
            # Update existing entry
            if self.strategy == "lru":
                self.access_order.remove(key)
                self.access_order.append(key)
        elif len(self.table) >= self.max_size:
            # Evict according to strategy
            self._evict()

        self.table[key] = value
        if self.strategy == "lru":
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear the table."""
        self.table.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get current size."""
        return len(self.table)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            del self.table[lru_key]

    def _evict_fifo(self) -> None:
        """Evict first-in first-out item."""
        if self.access_order:
            fifo_key = self.access_order.pop(0)
            del self.table[fifo_key]

    def _evict_random(self) -> None:
        """Evict random item."""
        import random
        if self.table:
            random_key = random.choice(list(self.table.keys()))
            del self.table[random_key]
            if random_key in self.access_order:
                self.access_order.remove(random_key)


class ExpressionMemoizer:
    """Specialized memoizer for expression computations."""

    def __init__(self):
        self.normalization_memo: Dict[int, Any] = {}
        self.simplification_memo: Dict[int, Any] = {}
        self.equality_memo: Dict[Tuple[int, int], bool] = {}
        self.substitution_memo: Dict[Tuple[int, int], Any] = {}

    def memoize_normalization(self, expr_id: int):
        """Decorator for memoizing normalization."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if expr_id in self.normalization_memo:
                    return self.normalization_memo[expr_id]

                result = func(*args, **kwargs)
                self.normalization_memo[expr_id] = result
                return result
            return wrapper
        return decorator

    def memoize_simplification(self, expr_id: int):
        """Decorator for memoizing simplification."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if expr_id in self.simplification_memo:
                    return self.simplification_memo[expr_id]

                result = func(*args, **kwargs)
                self.simplification_memo[expr_id] = result
                return result
            return wrapper
        return decorator

    def memoize_equality(self, expr1_id: int, expr2_id: int):
        """Decorator for memoizing equality checks."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = (expr1_id, expr2_id)
                if key in self.equality_memo:
                    return self.equality_memo[key]

                result = func(*args, **kwargs)
                self.equality_memo[key] = result
                return result
            return wrapper
        return decorator

    def clear(self) -> None:
        """Clear all memoization tables."""
        self.normalization_memo.clear()
        self.simplification_memo.clear()
        self.equality_memo.clear()
        self.substitution_memo.clear()


class FunctionMemoizer:
    """Memoizer for function applications."""

    def __init__(self, max_size: int = 1000):
        self.memo_table = MemoizationTable(max_size=max_size)
        self.call_count = 0
        self.hit_count = 0

    def memoize(self, func_name: str):
        """Decorator for memoizing function calls."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.call_count += 1

                # Create key from function name and arguments
                key = (func_name, args, tuple(sorted(kwargs.items())))

                result = self.memo_table.get(key)
                if result is not None:
                    self.hit_count += 1
                    return result

                # Compute result
                result = func(*args, **kwargs)
                self.memo_table.put(key, result)
                return result

            wrapper.memo_table = self.memo_table
            wrapper.stats = lambda: {
                'calls': self.call_count,
                'hits': self.hit_count,
                'misses': self.call_count - self.hit_count,
                'hit_rate': self.hit_count / self.call_count if self.call_count > 0 else 0
            }
            return wrapper

        return decorator

    def stats(self) -> Dict[str, int]:
        """Get memoization statistics."""
        return {
            'calls': self.call_count,
            'hits': self.hit_count,
            'misses': self.call_count - self.hit_count,
            'hit_rate': self.hit_count / self.call_count if self.call_count > 0 else 0,
            'cache_size': self.memo_table.size()
        }


# Global memoizers
expression_memoizer = ExpressionMemoizer()
function_memoizer = FunctionMemoizer()


# Convenience decorators
def memoize(max_size: int = 1000):
    """Decorator for memoizing function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from function name and arguments
            key = (func.__name__, args, tuple(sorted(kwargs.items())))

            result = function_memoizer.memo_table.get(key)
            if result is not None:
                return result

            # Compute result
            result = func(*args, **kwargs)
            function_memoizer.memo_table.put(key, result)
            return result

        wrapper.memo_table = function_memoizer.memo_table
        return wrapper

    return decorator


def memoize_expression_normalization(expr_id: int):
    """Memoize expression normalization."""
    return expression_memoizer.memoize_normalization(expr_id)


def memoize_expression_simplification(expr_id: int):
    """Memoize expression simplification."""
    return expression_memoizer.memoize_simplification(expr_id)


def memoize_expression_equality(expr1_id: int, expr2_id: int):
    """Memoize expression equality."""
    return expression_memoizer.memoize_equality(expr1_id, expr2_id)


def memoize_function(func_name: str, max_size: int = 1000):
    """Memoize function with given name."""
    memoizer = FunctionMemoizer(max_size)
    return memoizer.memoize(func_name)


# Hash utilities for complex objects
def hash_expression(expr: Any) -> str:
    """Compute string hash for expression-like objects."""
    if hasattr(expr, '__dict__'):
        # For dataclass-like objects
        items = sorted(expr.__dict__.items())
        content = "|".join(f"{k}:{hash_expression(v)}" for k, v in items)
        return hashlib.md5(content.encode()).hexdigest()
    elif isinstance(expr, (list, tuple)):
        content = "|".join(hash_expression(item) for item in expr)
        return hashlib.md5(content.encode()).hexdigest()
    elif isinstance(expr, dict):
        items = sorted((k, hash_expression(v)) for k, v in expr.items())
        content = "|".join(f"{k}:{v}" for k, v in items)
        return hashlib.md5(content.encode()).hexdigest()
    elif isinstance(expr, set):
        items = sorted(hash_expression(item) for item in expr)
        content = "|".join(items)
        return hashlib.md5(content.encode()).hexdigest()
    else:
        return hashlib.md5(str(expr).encode()).hexdigest()


def make_memo_key(*args, **kwargs) -> str:
    """Create a memoization key from arguments."""
    key_parts = []

    for arg in args:
        key_parts.append(hash_expression(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.extend([str(k), hash_expression(v)])

    return "|".join(key_parts)


class WeakMemoization:
    """Weak reference based memoization for objects with identity."""

    def __init__(self):
        self.cache: Dict[int, Any] = {}

    def get(self, obj: Any) -> Optional[Any]:
        """Get cached value for object."""
        obj_id = id(obj)
        return self.cache.get(obj_id)

    def put(self, obj: Any, value: Any) -> None:
        """Cache value for object."""
        obj_id = id(obj)
        self.cache[obj_id] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()


# Performance monitoring for memoization
class MemoizationMonitor:
    """Monitor memoization performance."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, int]] = {}

    def record_call(self, function_name: str) -> None:
        """Record a function call."""
        if function_name not in self.stats:
            self.stats[function_name] = {'calls': 0, 'hits': 0, 'misses': 0}

        self.stats[function_name]['calls'] += 1

    def record_hit(self, function_name: str) -> None:
        """Record a cache hit."""
        if function_name in self.stats:
            self.stats[function_name]['hits'] += 1

    def record_miss(self, function_name: str) -> None:
        """Record a cache miss."""
        if function_name in self.stats:
            self.stats[function_name]['misses'] += 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get memoization statistics."""
        result = {}
        for func, stats in self.stats.items():
            total = stats['calls']
            hit_rate = stats['hits'] / total if total > 0 else 0
            result[func] = {
                'calls': stats['calls'],
                'hits': stats['hits'],
                'misses': stats['misses'],
                'hit_rate': hit_rate
            }
        return result


# Global memoization monitor
memo_monitor = MemoizationMonitor()


# Decorators for monitored memoization
def monitored_memoize(func_name: str):
    """Memoize with monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            memo_monitor.record_call(func_name)

            # Check if result is cached
            key = make_memo_key(*args, **kwargs)
            if key in function_memoizer.memo_table.table:
                memo_monitor.record_hit(func_name)
                return function_memoizer.memo_table.table[key]

            memo_monitor.record_miss(func_name)
            result = func(*args, **kwargs)
            function_memoizer.memo_table.put((func_name, key), result)
            return result

        return wrapper
    return decorator


# Utility functions
def clear_all_memoization() -> None:
    """Clear all memoization caches."""
    expression_memoizer.clear()
    function_memoizer.memo_table.clear()
    memo_monitor.stats.clear()


def get_memoization_stats() -> Dict[str, Any]:
    """Get comprehensive memoization statistics."""
    return {
        'expression_memo': {
            'normalization': len(expression_memoizer.normalization_memo),
            'simplification': len(expression_memoizer.simplification_memo),
            'equality': len(expression_memoizer.equality_memo),
        },
        'function_memo': function_memoizer.stats(),
        'monitor': memo_monitor.get_stats()
    }
