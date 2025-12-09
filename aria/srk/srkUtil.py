"""
Common utility functions for SRK.

This module provides various utility functions including:
- Binary search in sorted arrays
- Array merging operations
- Pretty printing utilities
- Enumeration operations
- Integer set and map data structures
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Union, Any, Callable, TypeVar, Generic, Iterator
from collections import defaultdict
import io
import functools

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')


def binary_search(array: List[T], value: T, compare: Optional[Callable[[T, T], int]] = None) -> int:
    """
    Search for a value in a sorted array using binary search.

    Args:
        array: Sorted array to search in
        value: Value to search for
        compare: Comparison function (defaults to standard comparison)

    Returns:
        Index of the value if found

    Raises:
        ValueError: If value is not found
    """
    if compare is None:
        compare = lambda a, b: (a > b) - (a < b)

    left, right = 0, len(array) - 1

    while left <= right:
        mid = left + (right - left) // 2
        cmp = compare(value, array[mid])

        if cmp == 0:
            return mid
        elif cmp < 0:
            right = mid - 1
        else:
            left = mid + 1

    raise ValueError(f"Value {value} not found in array")


def merge_arrays(a: List[T], b: List[T], compare: Optional[Callable[[T, T], int]] = None) -> List[T]:
    """
    Merge two sorted arrays into a single sorted array.

    Args:
        a: First sorted array
        b: Second sorted array
        compare: Comparison function

    Returns:
        Merged sorted array
    """
    if compare is None:
        compare = lambda x, y: (x > y) - (x < y)

    result = []
    i = j = 0

    while i < len(a) and j < len(b):
        if compare(a[i], b[j]) <= 0:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

    # Add remaining elements
    result.extend(a[i:])
    result.extend(b[j:])

    return result


def exp(base: T, exponent: int, mul: Callable[[T, T], T], one: T) -> T:
    """
    Compute base^exponent using exponentiation by squaring.

    Args:
        base: Base value
        exponent: Exponent (non-negative integer)
        mul: Multiplication function
        one: Identity element for multiplication

    Returns:
        base^exponent
    """
    if exponent == 0:
        return one
    elif exponent == 1:
        return base
    else:
        half = exp(base, exponent // 2, mul, one)
        half_squared = mul(half, half)
        if exponent % 2 == 0:
            return half_squared
        else:
            return mul(base, half_squared)


def format_to_string(format_func: Callable, value: Any) -> str:
    """
    Format a value to string using a formatting function.

    Args:
        format_func: Function that takes a formatter and value
        value: Value to format

    Returns:
        Formatted string representation
    """
    # Simple implementation - in a full version this would use proper formatting
    return str(value)


def print_enum(formatter: Any, enum: Iterator[T], pp_elt: Callable, pp_sep: Optional[Callable] = None) -> None:
    """
    Pretty print an enumeration of elements.

    Args:
        formatter: Output formatter
        enum: Enumeration to print
        pp_elt: Function to print individual elements
        pp_sep: Function to print separators (defaults to comma-space)
    """
    if pp_sep is None:
        pp_sep = lambda f: print(", ", end="")

    elements = list(enum)
    if not elements:
        return

    pp_elt(elements[0])
    for elt in elements[1:]:
        pp_sep(formatter)
        pp_elt(elt)


def print_list(pp_elt: Callable, xs: List[T]) -> str:
    """
    Pretty print a list.

    Args:
        pp_elt: Function to print individual elements
        xs: List to print

    Returns:
        String representation of the list
    """
    if not xs:
        return "[]"

    # Simple implementation
    elements = [str(x) for x in xs]
    return f"[{', '.join(elements)}]"


def cartesian_product(e1: Iterator[T], e2: Iterator[U]) -> Iterator[Tuple[T, U]]:
    """
    Compute the Cartesian product of two enumerations.

    Args:
        e1: First enumeration
        e2: Second enumeration

    Yields:
        Pairs (x, y) where x is from e1 and y is from e2
    """
    e2_list = list(e2)
    for x in e1:
        for y in e2_list:
            yield (x, y)


def tuples(xss: List[Iterator[T]]) -> Iterator[List[T]]:
    """
    Generate all tuples from a list of enumerations.

    Args:
        xss: List of enumerations

    Yields:
        All possible combinations as lists
    """
    if not xss:
        yield []
        return

    for head in xss[0]:
        for tail in tuples(xss[1:]):
            yield [head] + tail


def adjacent_pairs(enum: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """
    Generate adjacent pairs from an enumeration.

    Args:
        enum: Input enumeration

    Yields:
        Pairs (x, y) where y is the next element after x
    """
    enum_list = list(enum)
    for i in range(len(enum_list) - 1):
        yield (enum_list[i], enum_list[i + 1])


def distinct_pairs(enum: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """
    Generate all distinct pairs from an enumeration.

    Args:
        enum: Input enumeration

    Yields:
        Pairs (x, y) where x comes before y in the enumeration
    """
    enum_list = list(enum)
    for i in range(len(enum_list)):
        for j in range(i + 1, len(enum_list)):
            yield (enum_list[i], enum_list[j])


class IntSet:
    """Set of integers with efficient operations."""

    def __init__(self, elements: Optional[Set[int]] = None):
        self._elements = elements or set()

    def add(self, x: int) -> None:
        """Add an element to the set."""
        self._elements.add(x)

    def remove(self, x: int) -> None:
        """Remove an element from the set."""
        self._elements.discard(x)

    def contains(self, x: int) -> bool:
        """Check if element is in the set."""
        return x in self._elements

    def union(self, other: IntSet) -> IntSet:
        """Union with another set."""
        return IntSet(self._elements | other._elements)

    def intersection(self, other: IntSet) -> IntSet:
        """Intersection with another set."""
        return IntSet(self._elements & other._elements)

    def difference(self, other: IntSet) -> IntSet:
        """Difference with another set."""
        return IntSet(self._elements - other._elements)

    def is_empty(self) -> bool:
        """Check if set is empty."""
        return len(self._elements) == 0

    def size(self) -> int:
        """Get size of the set."""
        return len(self._elements)

    def to_list(self) -> List[int]:
        """Convert to sorted list."""
        return sorted(list(self._elements))

    def __iter__(self):
        return iter(self._elements)

    def __str__(self) -> str:
        elements = sorted(list(self._elements))
        return f"{{{', '.join(map(str, elements))}}}"


class IntMap:
    """Map from integers to values."""

    def __init__(self, mapping: Optional[Dict[int, T]] = None):
        self._mapping = dict(mapping) if mapping else {}

    def get(self, key: int, default: Optional[T] = None) -> Optional[T]:
        """Get value for key."""
        return self._mapping.get(key, default)

    def set(self, key: int, value: T) -> None:
        """Set value for key."""
        self._mapping[key] = value

    def remove(self, key: int) -> None:
        """Remove key from map."""
        self._mapping.pop(key, None)

    def contains(self, key: int) -> bool:
        """Check if key exists."""
        return key in self._mapping

    def keys(self) -> Set[int]:
        """Get all keys."""
        return set(self._mapping.keys())

    def values(self) -> List[T]:
        """Get all values."""
        return list(self._mapping.values())

    def items(self) -> List[Tuple[int, T]]:
        """Get all key-value pairs."""
        return list(self._mapping.items())

    def is_empty(self) -> bool:
        """Check if map is empty."""
        return len(self._mapping) == 0

    def size(self) -> int:
        """Get size of the map."""
        return len(self._mapping)

    def __getitem__(self, key: int) -> T:
        return self._mapping[key]

    def __setitem__(self, key: int, value: T) -> None:
        self._mapping[key] = value

    def __contains__(self, key: int) -> bool:
        return key in self._mapping

    def __str__(self) -> str:
        items = [f"{k} -> {v}" for k, v in self._mapping.items()]
        return f"{{{', '.join(items)}}}"


# Convenience functions for common operations
def make_int_set(elements: List[int]) -> IntSet:
    """Create an IntSet from a list of elements."""
    return IntSet(set(elements))


def make_int_map(pairs: List[Tuple[int, T]]) -> IntMap[T]:
    """Create an IntMap from a list of key-value pairs."""
    return IntMap(dict(pairs))


# Default separator function for pretty printing
def default_separator() -> str:
    """Default separator for pretty printing."""
    return ", "


# Utility functions for working with enumerations
def enum_to_list(enum: Iterator[T]) -> List[T]:
    """Convert an enumeration to a list."""
    return list(enum)


def list_to_enum(xs: List[T]) -> Iterator[T]:
    """Convert a list to an enumeration."""
    return iter(xs)
