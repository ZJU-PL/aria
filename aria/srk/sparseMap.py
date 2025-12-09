"""
Sparse map data structure for SRK.

This module implements sparse maps that automatically maintain sparsity by
removing entries with zero values. This is useful for representing sparse
matrices and other sparse data structures efficiently.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Generic, TypeVar, Iterator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import operator

K = TypeVar('K')  # Key type (must be ordered/comparable)
V = TypeVar('V')  # Value type (must have zero element and equality)
T = TypeVar('T')  # Accumulator type for fold operation


class ZeroValue(Generic[V]):
    """Protocol for types that have a zero value."""

    @staticmethod
    def zero() -> V:
        """Return the zero value for this type."""
        raise NotImplementedError


class SparseMap(Generic[K, V]):
    """
    Sparse map implementation that automatically removes zero values.

    This map maintains sparsity by automatically removing entries when
    their values become zero, providing efficient storage for sparse data.
    """

    def __init__(self, mapping: Optional[Dict[K, V]] = None):
        """Initialize a sparse map."""
        self._map: Dict[K, V] = {}
        if mapping:
            for k, v in mapping.items():
                if not self._is_zero(v):
                    self._map[k] = v

    def _is_zero(self, value: V) -> bool:
        """Check if a value is considered zero."""
        # Try to call zero() method if it exists (for ZeroValue protocol)
        if hasattr(value, 'zero'):
            return value == value.zero()
        # Otherwise use standard equality with 0
        try:
            return value == 0
        except:
            return False

    def get(self, key: K) -> V:
        """Get the value for a key (returns zero if key not present)."""
        return self._map.get(key, self._get_zero())

    def set(self, key: K, value: V) -> SparseMap[K, V]:
        """Set the value for a key."""
        new_map = SparseMap(self._map.copy())
        if self._is_zero(value):
            new_map._map.pop(key, None)
        else:
            new_map._map[key] = value
        return new_map

    def _get_zero(self) -> V:
        """Get the zero value for this map's value type."""
        # This is a generic implementation - subclasses may override
        return 0  # type: ignore

    def is_zero(self) -> bool:
        """Check if this map is empty (contains only zeros)."""
        return len(self._map) == 0

    def merge(self, other: SparseMap[K, V], combine: Callable[[V, V], V]) -> SparseMap[K, V]:
        """Merge this map with another using a combination function."""
        result_map = self._map.copy()

        for key, other_value in other._map.items():
            if key in result_map:
                combined = combine(result_map[key], other_value)
                if self._is_zero(combined):
                    del result_map[key]
                else:
                    result_map[key] = combined
            else:
                if not self._is_zero(other_value):
                    result_map[key] = other_value

        return SparseMap(result_map)

    def modify(self, key: K, modifier: Callable[[V], V]) -> SparseMap[K, V]:
        """Modify the value at a key using a modifier function."""
        current_value = self.get(key)
        new_value = modifier(current_value)

        return self.set(key, new_value)

    def map(self, mapper: Callable[[K, V], V]) -> SparseMap[K, V]:
        """Apply a function to all key-value pairs."""
        result_map = {}
        for key, value in self._map.items():
            new_value = mapper(key, value)
            if not self._is_zero(new_value):
                result_map[key] = new_value

        return SparseMap(result_map)

    def extract(self, key: K) -> Tuple[V, SparseMap[K, V]]:
        """Extract the value for a key and return the remaining map."""
        value = self.get(key)
        remaining = self.set(key, self._get_zero())
        return value, remaining

    def enum(self) -> Iterator[Tuple[K, V]]:
        """Enumerate all non-zero key-value pairs."""
        return iter(self._map.items())

    def of_enum(self, enum: Iterator[Tuple[K, V]]) -> SparseMap[K, V]:
        """Create a sparse map from an enumeration of pairs."""
        result_map = {}
        for key, value in enum:
            if not self._is_zero(value):
                result_map[key] = value
        return SparseMap(result_map)

    def fold(self, folder: Callable[[K, V, T], T], initial: T) -> T:
        """Fold over all key-value pairs."""
        acc = initial
        for key, value in self._map.items():
            acc = folder(key, value, acc)
        return acc

    def min_support(self) -> Optional[Tuple[K, V]]:
        """Get the key-value pair with the smallest key."""
        if not self._map:
            return None
        return min(self._map.items(), key=lambda x: x[0])

    def pop(self) -> Tuple[Tuple[K, V], SparseMap[K, V]]:
        """Remove and return an arbitrary key-value pair."""
        if not self._map:
            raise ValueError("Cannot pop from empty sparse map")

        # Get first item (in insertion order for determinism)
        key = next(iter(self._map.keys()))
        value = self._map[key]
        remaining = self.set(key, self._get_zero())

        return (key, value), remaining

    def hash(self, hash_fn: Callable[[Tuple[K, V]], int]) -> int:
        """Compute a hash of the map contents."""
        # Simple hash based on sorted items
        items = sorted(self._map.items(), key=lambda x: x[0])
        return hash(tuple(hash_fn(item) for item in items))

    def compare(self, compare_fn: Callable[[V, V], int], other: SparseMap[K, V]) -> int:
        """Compare this map with another."""
        # Simple lexicographic comparison
        self_items = sorted(self._map.items(), key=lambda x: x[0])
        other_items = sorted(other._map.items(), key=lambda x: x[0])

        for (k1, v1), (k2, v2) in zip(self_items, other_items):
            if k1 != k2:
                return (k1 > k2) - (k1 < k2)  # Compare keys
            cmp = compare_fn(v1, v2)
            if cmp != 0:
                return cmp

        # All compared elements equal, compare lengths
        return (len(self._map) > len(other._map)) - (len(self._map) < len(other._map))

    def keys(self) -> Set[K]:
        """Get all keys in the map."""
        return set(self._map.keys())

    def values(self) -> List[V]:
        """Get all values in the map."""
        return list(self._map.values())

    def items(self) -> List[Tuple[K, V]]:
        """Get all key-value pairs."""
        return list(self._map.items())

    def size(self) -> int:
        """Get the number of non-zero entries."""
        return len(self._map)

    def is_empty(self) -> bool:
        """Check if the map is empty."""
        return len(self._map) == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseMap):
            return False
        return self._map == other._map

    def __str__(self) -> str:
        if not self._map:
            return "{}"

        items = [f"{k}: {v}" for k, v in sorted(self._map.items(), key=lambda x: x[0])]
        return f"{{{', '.join(items)}}}"

    def __getitem__(self, key: K) -> V:
        return self.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        # This modifies the map in place (not immutable)
        if self._is_zero(value):
            self._map.pop(key, None)
        else:
            self._map[key] = value

    def __contains__(self, key: K) -> bool:
        return key in self._map


# Convenience functions
def make_sparse_map() -> SparseMap[K, V]:
    """Create an empty sparse map."""
    return SparseMap()


def make_sparse_map_from_dict(mapping: Dict[K, V]) -> SparseMap[K, V]:
    """Create a sparse map from a dictionary."""
    return SparseMap(mapping)


def make_sparse_map_from_enum(enum: Iterator[Tuple[K, V]]) -> SparseMap[K, V]:
    """Create a sparse map from an enumeration of pairs."""
    sm = SparseMap()
    for key, value in enum:
        sm[key] = value
    return sm


# Specialized sparse map for common value types
class IntSparseMap(SparseMap[K, int]):
    """Sparse map specialized for integer values."""

    def _get_zero(self) -> int:
        return 0


class FloatSparseMap(SparseMap[K, float]):
    """Sparse map specialized for float values."""

    def _get_zero(self) -> float:
        return 0.0


# Utility functions for common operations
def sparse_map_add(sm1: SparseMap[K, int], sm2: SparseMap[K, int]) -> SparseMap[K, int]:
    """Add two integer sparse maps."""
    return sm1.merge(sm2, operator.add)


def sparse_map_multiply(sm1: SparseMap[K, int], sm2: SparseMap[K, int]) -> SparseMap[K, int]:
    """Multiply two integer sparse maps (element-wise)."""
    return sm1.merge(sm2, operator.mul)


def sparse_map_scale(sm: SparseMap[K, int], scalar: int) -> SparseMap[K, int]:
    """Scale a sparse map by a scalar."""
    return sm.map(lambda k, v: v * scalar)


# Example usage and testing
def test_sparse_map():
    """Test the sparse map implementation."""
    # Test basic operations
    sm = make_sparse_map()

    # Set some values
    sm[1] = 5
    sm[2] = 0  # Should be automatically removed
    sm[3] = 10

    assert sm.size() == 2
    assert sm.get(1) == 5
    assert sm.get(2) == 0  # Zero value
    assert sm.get(3) == 10
    assert sm.get(99) == 0  # Missing key

    # Test modification
    sm = sm.modify(1, lambda x: x * 2)
    assert sm.get(1) == 10

    # Test merge (addition)
    sm2 = make_sparse_map()
    sm2[1] = 3
    sm2[4] = 7

    sm_added = sm.merge(sm2, operator.add)
    assert sm_added.get(1) == 13  # 10 + 3
    assert sm_added.get(3) == 10  # Only in first map
    assert sm_added.get(4) == 7   # Only in second map

    print("Sparse map tests passed!")


if __name__ == "__main__":
    test_sparse_map()
