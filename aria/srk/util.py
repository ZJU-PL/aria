"""
Utility functions and helper classes for SRK.

This module provides essential utility functions and helper classes that support
the core symbolic reasoning functionality in SRK. It includes mathematical
utilities, data structures, and common algorithms used throughout the system.

Key Components:
- Integer utilities (ZZ) for basic arithmetic operations
- Rational number utilities (QQ) for fraction arithmetic
- Set and map operations for efficient data management
- Combinatorial utilities for algorithm implementation
- Input/output utilities for parsing and formatting
- Performance utilities for caching and memoization

Example:
    >>> from aria.srk.util import ZZ, QQ
    >>> print(ZZ.gcd(48, 18))  # 6
    >>> print(QQ.add(QQ.of_int(1, 2), QQ.of_int(1, 3)))  # 5/6
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable, TypeVar, Iterator,Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from contextlib import contextmanager, redirect_stdout
import math
import itertools
import bisect
from io import StringIO

T = TypeVar('T')
U = TypeVar('U')
K = TypeVar('K')
V = TypeVar('V')


class ZZ:
    """Integer utilities for SRK.

    This class provides static methods for common integer operations used
    throughout the symbolic reasoning system. All operations work with
    Python's built-in int type and handle edge cases appropriately.

    The class follows the mathematical convention where ZZ represents
    the integers (Zahlen in German), similar to how QQ represents
    rational numbers.
    """

    @staticmethod
    def one() -> int:
        """Return the integer 1.

        Returns:
            int: The integer one.
        """
        return 1

    @staticmethod
    def zero() -> int:
        """Return the integer 0.

        Returns:
            int: The integer zero.
        """
        return 0

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Compute the least common multiple of two integers.

        Args:
            a (int): First integer.
            b (int): Second integer.

        Returns:
            int: The LCM of a and b, or 0 if either input is 0.

        Example:
            >>> ZZ.lcm(12, 18)  # 36
            >>> ZZ.lcm(7, 5)   # 35
        """
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // math.gcd(a, b)

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Compute the greatest common divisor of two integers.

        Args:
            a (int): First integer.
            b (int): Second integer.

        Returns:
            int: The GCD of a and b (always non-negative).

        Example:
            >>> ZZ.gcd(48, 18)  # 6
            >>> ZZ.gcd(7, 5)   # 1
        """
        return math.gcd(abs(a), abs(b))

    @staticmethod
    def mul(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @staticmethod
    def sub(a: int, b: int) -> int:
        """Subtract two integers."""
        return a - b

    @staticmethod
    def negate(a: int) -> int:
        """Negate an integer."""
        return -a

    @staticmethod
    def abs(a: int) -> int:
        """Absolute value of an integer."""
        return abs(a)

    @staticmethod
    def equal(a: int, b: int) -> bool:
        """Check if two integers are equal."""
        return a == b

    @staticmethod
    def lt(a: int, b: int) -> bool:
        """Check if a < b."""
        return a < b

    @staticmethod
    def leq(a: int, b: int) -> bool:
        """Check if a <= b."""
        return a <= b

    @staticmethod
    def gt(a: int, b: int) -> bool:
        """Check if a > b."""
        return a > b

    @staticmethod
    def geq(a: int, b: int) -> bool:
        """Check if a >= b."""
        return a >= b

    @staticmethod
    def to_int(x: Fraction) -> Optional[int]:
        """Convert a fraction to integer if it's an integer value."""
        if x.denominator == 1:
            return int(x.numerator)
        return None

    @staticmethod
    def of_int(x: int) -> Fraction:
        """Convert an integer to a fraction."""
        return Fraction(x)

    @staticmethod
    def of_zz(x: int) -> Fraction:
        """Convert an integer to a fraction (alias for of_int)."""
        return Fraction(x)

    @staticmethod
    def to_zz(x: Fraction) -> Optional[int]:
        """Convert a fraction to integer if it's an integer value (alias for to_int)."""
        return ZZ.to_int(x)


class BatEnum:
    """Enumeration utilities for SRK (similar to OCaml's BatEnum)."""

    @staticmethod
    def empty() -> Iterator[Any]:
        """Return an empty enumeration."""
        return iter([])

    @staticmethod
    def singleton(x: T) -> Iterator[T]:
        """Return an enumeration with a single element."""
        return iter([x])

    @staticmethod
    def of_list(xs: List[T]) -> Iterator[T]:
        """Convert a list to an enumeration."""
        return iter(xs)

    @staticmethod
    def of_enum(enum: Iterator[T]) -> Iterator[T]:
        """Convert an enumeration to an enumeration (identity)."""
        return enum

    @staticmethod
    def fold(f: Callable[[U, T], U], init: U, enum: Iterator[T]) -> U:
        """Fold over an enumeration."""
        result = init
        for x in enum:
            result = f(result, x)
        return result

    @staticmethod
    def map(f: Callable[[T], U], enum: Iterator[T]) -> Iterator[U]:
        """Map over an enumeration."""
        return map(f, enum)

    @staticmethod
    def filter(f: Callable[[T], bool], enum: Iterator[T]) -> Iterator[T]:
        """Filter an enumeration."""
        return filter(f, enum)

    @staticmethod
    def take(n: int, enum: Iterator[T]) -> Iterator[T]:
        """Take first n elements from an enumeration."""
        count = 0
        for x in enum:
            if count >= n:
                break
            yield x
            count += 1

    @staticmethod
    def drop(n: int, enum: Iterator[T]) -> Iterator[T]:
        """Drop first n elements from an enumeration."""
        count = 0
        for x in enum:
            if count >= n:
                yield x
            count += 1

    @staticmethod
    def repeat(x: T, times: int = -1) -> Iterator[T]:
        """Repeat an element n times or infinitely."""
        if times == -1:
            while True:
                yield x
        else:
            for _ in range(times):
                yield x

    @staticmethod
    def range(start: int, stop: Optional[int] = None, step: int = 1) -> Iterator[int]:
        """Create a range enumeration."""
        if stop is None:
            return iter(range(start))
        else:
            return iter(range(start, stop, step))


class BatList:
    """List utilities for SRK (similar to OCaml's BatList)."""

    @staticmethod
    def of_enum(enum: Iterator[T]) -> List[T]:
        """Convert an enumeration to a list."""
        return list(enum)

    @staticmethod
    def fold_left(f: Callable[[U, T], U], init: U, xs: List[T]) -> U:
        """Left fold over a list."""
        result = init
        for x in xs:
            result = f(result, x)
        return result

    @staticmethod
    def fold_right(f: Callable[[T, U], U], xs: List[T], init: U) -> U:
        """Right fold over a list."""
        result = init
        for x in reversed(xs):
            result = f(x, result)
        return result

    @staticmethod
    def map(f: Callable[[T], U], xs: List[T]) -> List[U]:
        """Map over a list."""
        return [f(x) for x in xs]

    @staticmethod
    def filter(f: Callable[[T], bool], xs: List[T]) -> List[T]:
        """Filter a list."""
        return [x for x in xs if f(x)]

    @staticmethod
    def exists(f: Callable[[T], bool], xs: List[T]) -> bool:
        """Check if any element satisfies the predicate."""
        return any(f(x) for x in xs)

    @staticmethod
    def for_all(f: Callable[[T], bool], xs: List[T]) -> bool:
        """Check if all elements satisfy the predicate."""
        return all(f(x) for x in xs)

    @staticmethod
    def find(f: Callable[[T], bool], xs: List[T]) -> Optional[T]:
        """Find the first element satisfying the predicate."""
        for x in xs:
            if f(x):
                return x
        return None

    @staticmethod
    def partition(f: Callable[[T], bool], xs: List[T]) -> Tuple[List[T], List[T]]:
        """Partition a list into two lists based on a predicate."""
        yes = []
        no = []
        for x in xs:
            if f(x):
                yes.append(x)
            else:
                no.append(x)
        return yes, no


class BatSet:
    """Set utilities for SRK (similar to OCaml's BatSet)."""

    @staticmethod
    def of_list(xs: List[T]) -> Set[T]:
        """Convert a list to a set."""
        return set(xs)

    @staticmethod
    def add(s: Set[T], x: T) -> Set[T]:
        """Add an element to a set."""
        s.add(x)
        return s

    @staticmethod
    def remove(s: Set[T], x: T) -> Set[T]:
        """Remove an element from a set."""
        s.discard(x)
        return s

    @staticmethod
    def mem(s: Set[T], x: T) -> bool:
        """Check if an element is in a set."""
        return x in s

    @staticmethod
    def union(s1: Set[T], s2: Set[T]) -> Set[T]:
        """Union of two sets."""
        return s1 | s2

    @staticmethod
    def intersection(s1: Set[T], s2: Set[T]) -> Set[T]:
        """Intersection of two sets."""
        return s1 & s2

    @staticmethod
    def difference(s1: Set[T], s2: Set[T]) -> Set[T]:
        """Difference of two sets."""
        return s1 - s2

    @staticmethod
    def subset(s1: Set[T], s2: Set[T]) -> bool:
        """Check if s1 is a subset of s2."""
        return s1 <= s2


class BatMap:
    """Map utilities for SRK (similar to OCaml's BatMap)."""

    @staticmethod
    def empty() -> Dict[K, V]:
        """Create an empty map."""
        return {}

    @staticmethod
    def add(m: Dict[K, V], k: K, v: V) -> Dict[K, V]:
        """Add a key-value pair to a map."""
        m[k] = v
        return m

    @staticmethod
    def find(m: Dict[K, V], k: K) -> V:
        """Find the value associated with a key."""
        return m[k]

    @staticmethod
    def mem(m: Dict[K, V], k: K) -> bool:
        """Check if a key exists in a map."""
        return k in m

    @staticmethod
    def remove(m: Dict[K, V], k: K) -> Dict[K, V]:
        """Remove a key from a map."""
        if k in m:
            del m[k]
        return m


class BatHashtbl:
    """Hashtable utilities for SRK (similar to OCaml's Hashtbl)."""

    @staticmethod
    def create(size: int = 97) -> Dict[K, V]:
        """Create a new hashtable."""
        return {}

    @staticmethod
    def add(table: Dict[K, V], k: K, v: V) -> None:
        """Add a key-value pair to a hashtable."""
        table[k] = v

    @staticmethod
    def find(table: Dict[K, V], k: K) -> V:
        """Find the value associated with a key."""
        return table[k]

    @staticmethod
    def mem(table: Dict[K, V], k: K) -> bool:
        """Check if a key exists in a hashtable."""
        return k in table

    @staticmethod
    def replace(table: Dict[K, V], k: K, v: V) -> None:
        """Replace the value associated with a key."""
        table[k] = v


class BatDynArray:
    """Dynamic array utilities for SRK (similar to OCaml's DynArray)."""

    def __init__(self, elements: Optional[List[T]] = None):
        """Initialize a dynamic array."""
        self._data = elements[:] if elements else []

    def append(self, element: T) -> None:
        """Append an element to the array."""
        self._data.append(element)

    def get(self, index: int) -> T:
        """Get an element at the given index."""
        return self._data[index]

    def set(self, index: int, element: T) -> None:
        """Set an element at the given index."""
        if index >= len(self._data):
            self._data.extend([None] * (index - len(self._data) + 1))
        self._data[index] = element

    def length(self) -> int:
        """Get the length of the array."""
        return len(self._data)

    def to_list(self) -> List[T]:
        """Convert to a list."""
        return self._data[:]

    def clear(self) -> None:
        """Clear the array."""
        self._data.clear()

    def is_empty(self) -> bool:
        """Check if the array is empty."""
        return len(self._data) == 0

    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._data)

    def __getitem__(self, index: int) -> T:
        """Get an element at the given index."""
        return self._data[index]

    def __setitem__(self, index: int, element: T) -> None:
        """Set an element at the given index."""
        self.set(index, element)


class BatArray:
    """Array utilities for SRK (similar to OCaml's BatArray)."""

    @staticmethod
    def of_list(xs: List[T]) -> List[T]:
        """Convert a list to an array (just returns the list in Python)."""
        return xs

    @staticmethod
    def to_list(arr: List[T]) -> List[T]:
        """Convert an array to a list (just returns the array in Python)."""
        return arr

    @staticmethod
    def fold_left(f: Callable[[U, T], U], init: U, arr: List[T]) -> U:
        """Left fold over an array."""
        return BatList.fold_left(f, init, arr)

    @staticmethod
    def fold_right(f: Callable[[T, U], U], arr: List[T], init: U) -> U:
        """Right fold over an array."""
        return BatList.fold_right(f, arr, init)

    @staticmethod
    def map(f: Callable[[T], U], arr: List[T]) -> List[U]:
        """Map over an array."""
        return [f(x) for x in arr]

    @staticmethod
    def iter(f: Callable[[T], None], arr: List[T]) -> None:
        """Iterate over an array."""
        for x in arr:
            f(x)

T = TypeVar('T')
U = TypeVar('U')


class IntSet:
    """Set of integers with efficient operations."""

    def __init__(self, elements: Optional[Set[int]] = None):
        self.elements = elements or set()

    def add(self, element: int) -> None:
        """Add an element."""
        self.elements.add(element)

    def remove(self, element: int) -> None:
        """Remove an element."""
        self.elements.discard(element)

    def contains(self, element: int) -> bool:
        """Check if element is in set."""
        return element in self.elements

    def union(self, other: IntSet) -> IntSet:
        """Union with another set."""
        return IntSet(self.elements | other.elements)

    def intersection(self, other: IntSet) -> IntSet:
        """Intersection with another set."""
        return IntSet(self.elements & other.elements)

    def difference(self, other: IntSet) -> IntSet:
        """Difference with another set."""
        return IntSet(self.elements - other.elements)

    def is_empty(self) -> bool:
        """Check if set is empty."""
        return len(self.elements) == 0

    def size(self) -> int:
        """Get size of set."""
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __str__(self) -> str:
        return f"IntSet({sorted(self.elements)})"


class IntMap(dict):
    """Map from integers to values with dict-like interface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, key: int, value: Any) -> None:
        """Add a key-value pair."""
        self[key] = value

    def find(self, key: int) -> Any:
        """Find the value associated with a key."""
        return self[key]

    def mem(self, key: int) -> bool:
        """Check if a key exists."""
        return key in self

    def remove(self, key: int) -> None:
        """Remove a key."""
        if key in self:
            del self[key]


# Aliases for backward compatibility
BatMap = IntMap  # For backward compatibility with existing code


def make_int_set(elements: Optional[Set[int]] = None) -> IntSet:
    """Create an IntSet from a set of elements."""
    if elements is None:
        elements = set()
    return IntSet(elements)


def make_int_map() -> IntMap:
    """Create an empty IntMap."""
    return IntMap()


def binary_search(arr: List[int], target: int) -> int:
    """Binary search for target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # Not found


def merge_arrays(arr1: List[int], arr2: List[int]) -> List[int]:
    """Merge two sorted arrays."""
    result = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result


class Counter:
    """Counter for any hashable keys."""

    def __init__(self):
        self.counts: Dict[Any, int] = {}

    def inc(self, key: Any, amount: int = 1) -> None:
        """Increment counter for key."""
        self.counts[key] = self.counts.get(key, 0) + amount

    def decrement(self, key: Any, amount: int = 1) -> None:
        """Decrement counter for key."""
        self.counts[key] = self.counts.get(key, 0) - amount
        if self.counts[key] <= 0:
            del self.counts[key]

    def get(self, key: Any, default: int = 0) -> int:
        """Get count for key."""
        return self.counts.get(key, default)

    def keys(self) -> List[Any]:
        """Get all keys."""
        return list(self.counts.keys())

    def values(self) -> List[int]:
        """Get all values."""
        return list(self.counts.values())

    def items(self) -> List[Tuple[Any, int]]:
        """Get all key-value pairs."""
        return list(self.counts.items())

    def clear(self) -> None:
        """Clear the counter."""
        self.counts.clear()

    def __len__(self) -> int:
        """Get number of keys."""
        return len(self.counts)

    def __getitem__(self, key: Any) -> int:
        """Get count for key."""
        return self.counts[key]

    def __setitem__(self, key: Any, value: int) -> None:
        """Set count for key."""
        if value <= 0:
            self.counts.pop(key, None)
        else:
            self.counts[key] = value

    def __str__(self) -> str:
        return f"Counter({self.counts})"


class Stack:
    """Stack data structure."""

    def __init__(self):
        self.items: List[Any] = []

    def push(self, item: Any) -> None:
        """Push item onto stack."""
        self.items.append(item)

    def pop(self) -> Any:
        """Pop item from stack."""
        if not self.items:
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self) -> Any:
        """Peek at top item."""
        if not self.items:
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def empty(self) -> bool:
        """Check if stack is empty."""
        return len(self.items) == 0

    def is_empty(self) -> bool:
        """Check if stack is empty (alias for empty)."""
        return self.empty()

    def size(self) -> int:
        """Get stack size."""
        return len(self.items)

    def clear(self) -> None:
        """Clear the stack."""
        self.items.clear()

    def __str__(self) -> str:
        return f"Stack({self.items})"


class Queue:
    """Queue data structure."""

    def __init__(self):
        self.items: List[Any] = []

    def enqueue(self, item: Any) -> None:
        """Add item to queue."""
        self.items.append(item)

    def dequeue(self) -> Any:
        """Remove item from queue."""
        if not self.items:
            raise IndexError("dequeue from empty queue")
        return self.items.pop(0)

    def peek(self) -> Any:
        """Peek at front item."""
        if not self.items:
            raise IndexError("peek from empty queue")
        return self.items[0]

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.items) == 0

    def is_empty(self) -> bool:
        """Check if queue is empty (alias for empty)."""
        return self.empty()

    def size(self) -> int:
        """Get queue size."""
        return len(self.items)

    def clear(self) -> None:
        """Clear the queue."""
        self.items.clear()

    def __str__(self) -> str:
        return f"Queue({self.items})"


class PriorityQueue:
    """Priority queue implementation."""

    def __init__(self):
        self.items: List[Tuple[int, Any]] = []  # (priority, item)

    def insert(self, priority: int, item: Any) -> None:
        """Add item with priority."""
        # Insert in sorted order (lower priority number = higher priority)
        for i, (p, _) in enumerate(self.items):
            if priority < p:
                self.items.insert(i, (priority, item))
                return
        self.items.append((priority, item))

    def enqueue(self, item: Any, priority: int) -> None:
        """Add item with priority (alias for insert)."""
        self.insert(priority, item)

    def dequeue(self) -> Any:
        """Remove highest priority item."""
        if not self.items:
            raise IndexError("dequeue from empty priority queue")
        return self.items.pop(0)[1]

    def extract_min(self) -> Tuple[int, Any]:
        """Remove and return the item with minimum priority."""
        if not self.items:
            raise IndexError("extract_min from empty priority queue")
        return self.items.pop(0)

    def peek(self) -> Any:
        """Peek at highest priority item."""
        if not self.items:
            raise IndexError("peek from empty priority queue")
        return self.items[0][1]

    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.items) == 0

    def is_empty(self) -> bool:
        """Check if queue is empty (alias for empty)."""
        return self.empty()

    def size(self) -> int:
        """Get queue size."""
        return len(self.items)

    def clear(self) -> None:
        """Clear the queue."""
        self.items.clear()

    def __str__(self) -> str:
        return f"PriorityQueue({[(p, item) for p, item in self.items]})"


class Graph:
    """Simple graph data structure."""

    def __init__(self):
        self.vertices: Set[int] = set()
        self.edges: Dict[int, Set[int]] = {}

    def add_vertex(self, vertex: int) -> None:
        """Add a vertex."""
        self.vertices.add(vertex)
        if vertex not in self.edges:
            self.edges[vertex] = set()

    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        """Add an edge."""
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        self.edges[from_vertex].add(to_vertex)

    def remove_edge(self, from_vertex: int, to_vertex: int) -> None:
        """Remove an edge."""
        if from_vertex in self.edges:
            self.edges[from_vertex].discard(to_vertex)

    def neighbors(self, vertex: int) -> Set[int]:
        """Get neighbors of vertex."""
        return self.edges.get(vertex, set())

    def degree(self, vertex: int) -> int:
        """Get degree of vertex."""
        return len(self.neighbors(vertex))

    def is_connected(self, vertex1: int, vertex2: int) -> bool:
        """Check if two vertices are connected."""
        return vertex2 in self.neighbors(vertex1)

    def __str__(self) -> str:
        edges_str = []
        for v in sorted(self.vertices):
            for neighbor in sorted(self.neighbors(v)):
                edges_str.append(f"{v}->{neighbor}")
        return f"Graph({', '.join(edges_str)})"


# Mathematical utility functions
def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b != 0:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """Compute least common multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def factorial(n: int) -> int:
    """Compute factorial."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def is_prime(n: int) -> bool:
    """Check if number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# Iterator utilities
def powerset(iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    """Generate all subsets of an iterable."""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def combinations_with_replacement(iterable: Iterable[T], r: int) -> Iterator[Tuple[T, ...]]:
    """Generate combinations with replacement."""
    return itertools.combinations_with_replacement(iterable, r)


def permutations(iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    """Generate all permutations of an iterable."""
    return itertools.permutations(iterable)


# String utilities
def indent_string(s: str, indent: str = "  ") -> str:
    """Indent each line of a string."""
    lines = s.split('\n')
    return '\n'.join(indent + line if line else '' for line in lines)


def format_number(n: Union[int, float, Fraction]) -> str:
    """Format a number nicely."""
    if isinstance(n, Fraction):
        if n.denominator == 1:
            return str(n.numerator)
        else:
            return f"{n.numerator}/{n.denominator}"
    elif isinstance(n, float):
        if n.is_integer():
            return str(int(n))
        else:
            return str(n)
    else:
        return str(n)


# Type checking utilities
def is_integer(x: Any) -> bool:
    """Check if value is an integer."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_rational(x: Any) -> bool:
    """Check if value is a rational number."""
    return isinstance(x, (int, Fraction))


def is_number(x: Any) -> bool:
    """Check if value is a number."""
    return isinstance(x, (int, float, Fraction))


def safe_divide(a: Union[int, float, Fraction], b: Union[int, float, Fraction]) -> Union[int, float, Fraction]:
    """Safe division that handles zero denominator."""
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a / b


# Collection utilities
def flatten(nested_list: List[List[T]]) -> List[T]:
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


def unique(items: List[T]) -> List[T]:
    """Get unique items while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def group_by(items: List[T], key_func: Callable[[T], U]) -> Dict[U, List[T]]:
    """Group items by key function."""
    result = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def partition(items: List[T], predicate: Callable[[T], bool]) -> Tuple[List[T], List[T]]:
    """Partition items based on predicate."""
    true_items = []
    false_items = []
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    return true_items, false_items


# Error handling utilities
class SRKError(Exception):
    """Base exception for SRK errors."""
    pass


class TypeError(SRKError):
    """Type-related error."""
    pass


class ValueError(SRKError):
    """Value-related error."""
    pass


class NotImplementedError(SRKError):
    """Feature not implemented error."""
    pass


# Debugging utilities
def debug_print(obj: Any, label: str = "") -> None:
    """Debug print with optional label."""
    if label:
        print(f"{label}: {obj}")
    else:
        print(obj)


def debug_vars(**kwargs) -> None:
    """Debug print multiple variables."""
    for name, value in kwargs.items():
        print(f"{name} = {value}")


# Timing utilities
class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Start the timer."""
        import time
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        import time
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.time()
        return self.end_time - self.start_time

    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        import time
        if self.start_time is None:
            return 0.0

        current_time = time.time()
        return current_time - self.start_time

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None


@contextmanager
def time_block(label: str = "block"):
    """Context manager for timing code blocks."""
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        print(f"{label}: {elapsed:.4f}s")


# Memory usage utilities (simplified)
def estimate_size(obj: Any) -> int:
    """Estimate memory usage of object (simplified)."""
    import sys

    # Basic estimation - in practice would be more sophisticated
    if hasattr(obj, '__sizeof__'):
        return sys.getsizeof(obj)
    else:
        return 0


def format_bytes(bytes: int) -> str:
    """Format bytes in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"


# File and I/O utilities
def read_file_lines(filename: str) -> List[str]:
    """Read lines from file."""
    with open(filename, 'r') as f:
        return f.read().strip().split('\n')


def write_file_lines(filename: str, lines: List[str]) -> None:
    """Write lines to file."""
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


# Statistical utilities
def mean(values: List[float]) -> float:
    """Compute mean of values."""
    if not values:
        raise ValueError("Empty list")
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    """Compute median of values."""
    if not values:
        raise ValueError("Empty list")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]


def variance(values: List[float]) -> float:
    """Compute variance of values."""
    if not values:
        raise ValueError("Empty list")

    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def standard_deviation(values: List[float]) -> float:
    """Compute standard deviation of values."""
    return math.sqrt(variance(values))


# Random utilities
def random_int(a: int, b: int) -> int:
    """Generate random integer between a and b (inclusive)."""
    import random
    return random.randint(a, b)


def random_choice(items: List[T]) -> T:
    """Choose random item from list."""
    import random
    return random.choice(items)


def shuffle_list(items: List[T]) -> List[T]:
    """Shuffle a list in place and return it."""
    import random
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled


# Binary search utility
def search(value: T, array: List[T], compare: Optional[Callable[[T, T], int]] = None) -> int:
    """Search for an index in a sorted array using binary search."""
    if compare is None:
        compare = lambda a, b: (a > b) - (a < b)  # Default comparison

    def binary_search(min_idx: int, max_idx: int) -> int:
        if max_idx < min_idx:
            raise ValueError("Value not found")

        mid = min_idx + ((max_idx - min_idx) // 2)
        cmp = compare(value, array[mid])

        if cmp < 0:
            return binary_search(min_idx, mid - 1)
        elif cmp > 0:
            return binary_search(mid + 1, max_idx)
        else:
            return mid

    return binary_search(0, len(array) - 1)


# Array merging utility
def merge_array(a: List[T], b: List[T], compare: Optional[Callable[[T, T], int]] = None) -> List[T]:
    """Merge two sorted arrays into a single sorted array."""
    if compare is None:
        compare = lambda x, y: (x > y) - (x < y)

    alen = len(a)
    blen = len(b)

    # Count intersection size
    def count_common(i: int, j: int, acc: int) -> int:
        if i == alen or j == blen:
            return acc
        cmp = compare(a[i], b[j])
        if cmp < 0:
            return count_common(i + 1, j, acc)
        elif cmp > 0:
            return count_common(i, j + 1, acc)
        else:
            return count_common(i + 1, j + 1, acc + 1)

    clen = alen + blen - count_common(0, 0, 0)
    c = [None] * clen  # type: ignore

    def merge(i: int, j: int, k: int) -> None:
        if k < clen:
            if i == alen:
                c[k] = b[j]
                merge(i, j + 1, k + 1)
            elif j == blen:
                c[k] = a[i]
                merge(i + 1, j, k + 1)
            else:
                cmp = compare(a[i], b[j])
                if cmp < 0:
                    c[k] = a[i]
                    merge(i + 1, j, k + 1)
                elif cmp > 0:
                    c[k] = b[j]
                    merge(i, j + 1, k + 1)
                else:
                    c[k] = a[i]
                    merge(i + 1, j + 1, k + 1)

    merge(0, 0, 0)
    return c  # type: ignore


# Exponentiation utility
def exp(mul: Callable[[T, T], T], one: T, base: T, exponent: int) -> T:
    """Compute base^exponent using exponentiation by squaring."""
    if exponent == 0:
        return one
    elif exponent == 1:
        return base
    else:
        half = exp(mul, one, base, exponent // 2)
        squared = mul(half, half)
        if exponent % 2 == 0:
            return squared
        else:
            return mul(base, squared)


# Pretty printing utilities
def mk_show(pp_func: Callable, x: T) -> str:
    """Convert a pretty-printable object to string."""
    buffer = StringIO()
    pp_func(buffer, x)
    return buffer.getvalue()


def default_sep(formatter) -> None:
    """Default separator for pretty printing."""
    print(", ", file=formatter, end="")


def pp_print_enum_nobox(pp_elt: Callable, formatter, enum: Iterator[T], pp_sep: Optional[Callable] = None) -> None:
    """Pretty print enumeration without box."""
    if pp_sep is None:
        pp_sep = lambda f, _: default_sep(f)

    try:
        first = next(enum)
        pp_elt(formatter, first)
        for elt in enum:
            pp_sep(formatter, None)
            pp_elt(formatter, elt)
    except StopIteration:
        pass


def pp_print_enum(pp_elt: Callable, formatter, enum: Iterator[T], indent: int = 2, pp_sep: Optional[Callable] = None) -> None:
    """Pretty print enumeration with box."""
    if pp_sep is None:
        pp_sep = lambda f, _: default_sep(f)

    # Note: This is a simplified implementation since Python doesn't have Format.pp_open_hovbox
    pp_print_enum_nobox(pp_elt, formatter, enum, pp_sep)


# Enumeration utilities
def cartesian_product(e1: Iterator[T], e2: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """Cartesian product of two enumerations."""
    e2_list = list(e2)  # Convert to list to allow multiple iterations
    for x in e1:
        for y in e2_list:
            yield (x, y)


def tuples(enums: List[Iterator[T]]) -> Iterator[List[T]]:
    """Generate tuples from list of enumerations."""
    if not enums:
        yield []
        return

    if len(enums) == 1:
        for elt in enums[0]:
            yield [elt]
        return

    for head in enums[0]:
        for tail in tuples(enums[1:]):
            yield [head] + tail


def adjacent_pairs(enum: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """Generate adjacent pairs from enumeration."""
    enum_list = list(enum)
    for i in range(len(enum_list) - 1):
        yield (enum_list[i], enum_list[i + 1])


def distinct_pairs(enum: Iterator[T]) -> Iterator[Tuple[T, T]]:
    """Generate distinct pairs from enumeration."""
    enum_list = list(enum)
    for i in range(len(enum_list)):
        for j in range(i + 1, len(enum_list)):
            yield (enum_list[i], enum_list[j])


# List pretty printing
def pp_print_list(pp_elt: Callable, formatter, xs: List[T]) -> None:
    """Pretty print a list."""
    def sep(f, _) -> None:
        print("; ", file=f, end="")

    print("[", file=formatter, end="")
    pp_print_enum(pp_elt, formatter, iter(xs), pp_sep=sep)
    print("]", file=formatter, end="")
