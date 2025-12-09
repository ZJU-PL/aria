"""
Disjoint set (union-find) data structure for SRK.

This module implements an efficient disjoint set data structure with:
- Path compression optimization
- Union by rank optimization
- Generic element type support
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Generic, TypeVar, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

T = TypeVar('T')


class HashableType(Protocol):
    """Protocol for types that can be used as disjoint set elements."""

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...


@dataclass
class DisjointSetElement:
    """Internal representation of a disjoint set element."""

    id: int
    parent: Optional[DisjointSetElement] = None
    rank: int = 0


class DisjointSet(Generic[T]):
    """
    Disjoint set data structure with path compression and union by rank.

    This implementation uses a hash table to map elements to their
    internal set representations, allowing for efficient find and union operations.
    """

    def __init__(self, initial_capacity: int = 16):
        """Initialize a new disjoint set."""
        self._element_map: Dict[T, DisjointSetElement] = {}
        self._next_id: int = 0
        self._set_count: int = 0

    def copy(self) -> DisjointSet[T]:
        """Create a copy of this disjoint set."""
        new_ds = DisjointSet[T]()
        new_ds._element_map = self._element_map.copy()
        new_ds._next_id = self._next_id
        new_ds._set_count = self._set_count
        return new_ds

    def _make_set(self, element: T) -> DisjointSetElement:
        """Create a new singleton set containing the element."""
        if element in self._element_map:
            return self._element_map[element]

        # Create new element representation
        elem = DisjointSetElement(id=self._next_id, parent=None, rank=0)
        self._element_map[element] = elem
        self._next_id += 1
        self._set_count += 1
        return elem

    def _find_impl(self, element: DisjointSetElement) -> DisjointSetElement:
        """Find the root of the set containing element (with path compression)."""
        if element.parent is None:
            return element

        # Path compression: make all nodes on path point directly to root
        root = self._find_impl(element.parent)
        element.parent = root
        return root

    def find(self, element: T) -> DisjointSetElement:
        """
        Find the representative element of the set containing the given element.

        This operation implements path compression for efficiency.
        If the element is not in any set, it creates a new singleton set.
        """
        if element not in self._element_map:
            self._make_set(element)
        elem = self._element_map[element]
        return self._find_impl(elem)

    def union(self, x: T, y: T) -> None:
        """
        Union the sets containing elements x and y.

        This operation implements union by rank for efficiency.
        """
        x_elem = self._make_set(x)
        y_elem = self._make_set(y)

        x_root = self._find_impl(x_elem)
        y_root = self._find_impl(y_elem)

        if x_root.id == y_root.id:
            return  # Already in same set

        # Union by rank: attach smaller rank tree under root of higher rank tree
        if x_root.rank > y_root.rank:
            y_root.parent = x_root
        elif x_root.rank < y_root.rank:
            x_root.parent = y_root
        else:
            # Same rank: arbitrarily choose x_root as parent and increment its rank
            y_root.parent = x_root
            x_root.rank += 1

        self._set_count -= 1

    def same_set(self, x: T, y: T) -> bool:
        """Check if two elements are in the same set."""
        if x not in self._element_map or y not in self._element_map:
            return False
        return self.find(x).id == self.find(y).id

    def set_count(self) -> int:
        """Get the number of disjoint sets."""
        return self._set_count

    def size(self) -> int:
        """Get the total number of elements in all sets."""
        return len(self._element_map)

    def get_sets(self) -> List[Set[T]]:
        """
        Get all disjoint sets as a list of element sets.

        Returns:
            List where each element is a set of elements in one disjoint set
        """
        # Group elements by their root
        sets: Dict[int, Set[T]] = {}
        for element in self._element_map:
            root = self.find(element)
            if root.id not in sets:
                sets[root.id] = set()
            sets[root.id].add(element)

        return list(sets.values())

    def get_representatives(self) -> List[T]:
        """Get one representative element from each disjoint set."""
        representatives = []
        seen_roots = set()

        for element in self._element_map:
            root = self.find(element)
            if root.id not in seen_roots:
                representatives.append(element)
                seen_roots.add(root.id)

        return representatives

    def __str__(self) -> str:
        sets = self.get_sets()
        set_strs = [f"{{{', '.join(map(str, s))}}}" for s in sets]
        return f"DisjointSet([{', '.join(set_strs)}])"


# Convenience functions
def make_disjoint_set() -> DisjointSet[T]:
    """Create a new empty disjoint set."""
    return DisjointSet()


def make_disjoint_set_from_elements(elements: List[T]) -> DisjointSet[T]:
    """Create a disjoint set with the given elements as singleton sets."""
    ds = DisjointSet()
    for elem in elements:
        ds._make_set(elem)
    return ds


# Example usage and testing functions
def test_disjoint_set():
    """Test the disjoint set implementation."""
    ds = make_disjoint_set()

    # Add some elements
    elements = [1, 2, 3, 4, 5]

    # Initially all are in different sets
    assert ds.set_count() == 0

    # Find operations should create singleton sets
    ds.find(1)
    ds.find(2)
    assert ds.set_count() == 2

    # Union operations
    ds.union(1, 2)
    assert ds.set_count() == 1
    assert ds.same_set(1, 2)

    ds.union(3, 4)
    assert ds.set_count() == 2

    ds.union(2, 3)
    assert ds.set_count() == 1
    assert ds.same_set(1, 4)

    print("Disjoint set tests passed!")


if __name__ == "__main__":
    test_disjoint_set()
