"""
Caching and memoization utilities for performance optimization.

This module provides efficient caching mechanisms for expensive computations
that are frequently used throughout the symbolic reasoning system. It includes
LRU caches, function memoization decorators, and utilities for cache management.

Key Features:
- LRU (Least Recently Used) cache implementation
- Function memoization decorators for automatic caching
- Configurable cache sizes and eviction policies
- Thread-safe operations for concurrent environments
- Hash computation for complex key types
- Cache statistics and introspection utilities

Example:
    >>> from aria.utils.srk.structures.cache import LRUCache, memoize
    >>> # Direct cache usage
    >>> cache = LRUCache(max_size=100)
    >>> cache.put("key", expensive_computation())
    >>> value = cache.get("key")
    >>>
    >>> # Function memoization
    >>> @memoize(max_size=50)
    ... def fibonacci(n):
    ...     return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
"""

from __future__ import annotations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Any,
    Callable,
    TypeVar,
    Generic,
)
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import weakref

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Least Recently Used (LRU) cache implementation.

    This cache stores key-value pairs and automatically evicts the least
    recently used items when the maximum size is reached. It maintains
    an access order list to track which items were used most recently.

    The cache is particularly useful for caching expensive computations
    where recently used results are likely to be reused soon.

    Attributes:
        max_size (int): Maximum number of items to store before eviction.
        cache (Dict[K, V]): Internal storage mapping keys to values.
        access_order (List[K]): List tracking access order (most recent at end).

    Example:
        >>> cache = LRUCache(max_size=3)
        >>> cache.put("a", 1)
        >>> cache.put("b", 2)
        >>> cache.put("c", 3)
        >>> cache.get("a")  # Access moves "a" to end
        >>> cache.put("d", 4)  # Evicts "b" (least recently used)
    """

    def __init__(self, max_size: int = 128):
        """Initialize LRU cache with maximum size.

        Args:
            max_size: Maximum number of items to store. Must be positive.
        """
        self.max_size = max_size
        self.cache: Dict[K, V] = {}
        self.access_order: List[K] = []

    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: K, value: V) -> None:
        """Put value in cache."""
        if key in self.cache:
            # Update existing entry
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def __contains__(self, key: K) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


class WeakKeyCache(Generic[K, V]):
    """Cache using weak references for keys."""

    def __init__(self):
        self.cache: Dict[int, V] = {}

    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        key_id = id(key)
        return self.cache.get(key_id)

    def put(self, key: K, value: V) -> None:
        """Put value in cache."""
        key_id = id(key)
        self.cache[key_id] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class Memoize:
    """Decorator for memoizing function results."""

    def __init__(self, cache_factory: Callable[[], Dict] = None):
        self.cache_factory = cache_factory or (lambda: {})

    def __call__(self, func: Callable) -> Callable:
        cache = self.cache_factory()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from arguments
            key = self._make_key(args, kwargs)

            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            return result

        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper

    def _make_key(self, args: Tuple, kwargs: Dict) -> Any:
        """Create a hashable key from arguments."""
        # Convert arguments to a hashable form
        key_parts = []

        # Handle positional arguments
        for arg in args[1:]:  # Skip 'self' for methods
            key_parts.append(self._make_hashable(arg))

        # Handle keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.extend([k, self._make_hashable(v)])

        return tuple(key_parts)

    def _make_hashable(self, obj: Any) -> Any:
        """Convert object to hashable form."""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, set):
            return frozenset(self._make_hashable(item) for item in obj)
        elif hasattr(obj, "__dict__"):
            # For objects with __dict__, use their attributes
            return tuple(
                sorted((k, self._make_hashable(v)) for k, v in obj.__dict__.items())
            )
        else:
            # Fallback: use string representation
            return str(obj)


class FunctionCache:
    """Cache for storing results of function applications."""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[Tuple, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, func_name: str, args: Tuple) -> Optional[Any]:
        """Get cached result."""
        key = (func_name, args)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, func_name: str, args: Tuple, result: Any) -> None:
        """Cache a result."""
        key = (func_name, args)

        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class ExpressionCache:
    """Specialized cache for expression-related computations."""

    def __init__(self):
        self._normalization_cache: Dict[int, Any] = {}
        self._simplification_cache: Dict[int, Any] = {}
        self._equality_cache: Dict[Tuple[int, int], bool] = {}

    def get_normal_form(self, expr_id: int) -> Optional[Any]:
        """Get cached normal form."""
        return self._normalization_cache.get(expr_id)

    def put_normal_form(self, expr_id: int, normal_form: Any) -> None:
        """Cache normal form."""
        self._normalization_cache[expr_id] = normal_form

    def get_simplification(self, expr_id: int) -> Optional[Any]:
        """Get cached simplification."""
        return self._simplification_cache.get(expr_id)

    def put_simplification(self, expr_id: int, simplified: Any) -> None:
        """Cache simplification."""
        self._simplification_cache[expr_id] = simplified

    def get_equality(self, expr1_id: int, expr2_id: int) -> Optional[bool]:
        """Get cached equality check."""
        return self._equality_cache.get((expr1_id, expr2_id))

    def put_equality(self, expr1_id: int, expr2_id: int, equal: bool) -> None:
        """Cache equality check."""
        self._equality_cache[(expr1_id, expr2_id)] = equal

    def clear(self) -> None:
        """Clear all caches."""
        self._normalization_cache.clear()
        self._simplification_cache.clear()
        self._equality_cache.clear()


# Global cache instances
global_lru_cache = LRUCache()
global_function_cache = FunctionCache()
global_expression_cache = ExpressionCache()


# Convenience decorators
def memoize_lru(max_size: int = 128):
    """Decorator for LRU memoization."""
    cache = LRUCache(max_size)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.put(key, result)
            return result

        return wrapper

    return decorator


def memoize_weak():
    """Decorator for weak reference memoization."""
    cache = WeakKeyCache()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use first argument as key for weak caching
            if args:
                key = args[0]
                result = cache.get(key)
                if result is None:
                    result = func(*args, **kwargs)
                    cache.put(key, result)
                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def clear_caches() -> None:
    """Clear all global caches."""
    global_lru_cache.clear()
    global_function_cache.clear()
    global_expression_cache.clear()


# Utility functions for hashing complex objects
def hash_expression(expr: Any) -> int:
    """Compute hash for expression-like objects."""
    if hasattr(expr, "__dict__"):
        # For dataclass-like objects
        items = sorted(expr.__dict__.items())
        return hash(tuple((k, hash_expression(v)) for k, v in items))
    elif isinstance(expr, (list, tuple)):
        return hash(tuple(hash_expression(item) for item in expr))
    elif isinstance(expr, dict):
        return hash(tuple(sorted((k, hash_expression(v)) for k, v in expr.items())))
    elif isinstance(expr, set):
        return hash(frozenset(hash_expression(item) for item in expr))
    else:
        return hash(expr)


def make_cache_key(*args, **kwargs) -> Tuple:
    """Create a cache key from arguments."""
    key_parts = []

    for arg in args:
        key_parts.append(hash_expression(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.extend([k, hash_expression(v)])

    return tuple(key_parts)


# ---------------------------------------------------------------------------
# Missing OCaml API: cache parameters and additional operations
# ---------------------------------------------------------------------------

@dataclass
class CacheParams:
    """Cache configuration parameters (mirrors OCaml ``Cache.S.params``)."""
    max_size: int = 1000
    hard_limit: int = 2000
    keys_hit_rate: float = 0.5
    min_hits: int = 2
    aging_factor: float = 0.9


class LRUCache:
    """LRU cache with configurable parameters (mirrors OCaml ``Cache.LRU.S``).

    Extends Python's built-in LRU capabilities with parameter-awareness,
    reset, copy, and iteration operations.
    """

    def __init__(self, params: Optional[CacheParams] = None, size: Optional[int] = None):
        from collections import OrderedDict

        self._params = params or CacheParams()
        self._data: "OrderedDict" = OrderedDict()
        self._max_size = size or self._params.max_size

    def get_params(self) -> CacheParams:
        """Get current cache parameters."""
        return self._params

    def set_params(self, params: CacheParams) -> None:
        """Update cache parameters."""
        self._params = params
        self._max_size = self._params.max_size
        self._evict_if_needed()

    def reset(self) -> None:
        """Clear all entries from the cache."""
        self._data.clear()

    def copy(self) -> "LRUCache":
        """Create a shallow copy of the cache."""
        new_cache = LRUCache(params=self._params, size=self._max_size)
        new_cache._data.update(self._data)
        return new_cache

    def add(self, key: Any, value: Any) -> None:
        """Add or update a cache entry."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        self._evict_if_needed()

    def find(self, key: Any) -> Any:
        """Find a value by key. Raises KeyError if not found."""
        if key not in self._data:
            raise KeyError(key)
        self._data.move_to_end(key)
        return self._data[key]

    def mem(self, key: Any) -> bool:
        """Check if a key exists in the cache."""
        return key in self._data

    def remove(self, key: Any) -> None:
        """Remove a key from the cache."""
        self._data.pop(key, None)

    def iter(self, fn: Callable[[Any, Any], None]) -> None:
        """Iterate over all (key, value) pairs."""
        for k, v in self._data.items():
            fn(k, v)

    def fold(self, fn: Callable[[Any, Any, Any], Any], init: Any) -> Any:
        """Fold over all (key, value) pairs."""
        acc = init
        for k, v in self._data.items():
            acc = fn(k, v, acc)
        return acc

    def filter_map_inplace(self, fn: Callable[[Any, Any], Optional[Any]]) -> None:
        """Filter and map cache entries in-place."""
        new_data = type(self._data)()
        for k, v in self._data.items():
            result = fn(k, v)
            if result is not None:
                new_data[k] = result
        self._data = new_data

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
