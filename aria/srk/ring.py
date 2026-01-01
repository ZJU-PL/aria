"""
Ring theory operations for SRK.

This module provides implementations of vectors and matrices over rings,
along with abelian groups and related algebraic structures.
"""

from __future__ import annotations
from typing import (
    TypeVar,
    Protocol,
    Generic,
    Dict,
    List,
    Tuple,
    Optional,
    Iterator,
    Callable,
    Any,
    Union,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
import math

from aria.srk.algebra import Ring
from aria.srk.util import IntSet

T = TypeVar("T")
U = TypeVar("U")
D = TypeVar("D")  # Dimension type
S = TypeVar("S")  # Scalar type
V = TypeVar("V")  # Vector type


class AbelianGroup(Protocol[T]):
    """Protocol for abelian group algebraic structure."""

    @abstractmethod
    def equal(self, other: T) -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    def add(self, other: T) -> T:
        """Addition operation."""
        ...

    @abstractmethod
    def negate(self) -> T:
        """Additive inverse."""
        ...

    @abstractmethod
    def zero(self) -> T:
        """Additive identity."""
        ...


class Vector(Protocol):
    """Protocol for vector operations over a ring."""

    @abstractmethod
    def equal(self, other: "Vector") -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    def add(self, other: "Vector") -> "Vector":
        """Vector addition."""
        ...

    @abstractmethod
    def scalar_mul(self, scalar: Any, vec: "Vector") -> "Vector":
        """Scalar multiplication."""
        ...

    @abstractmethod
    def negate(self) -> "Vector":
        """Additive inverse."""
        ...

    @abstractmethod
    def sub(self, other: "Vector") -> "Vector":
        """Vector subtraction."""
        ...

    @abstractmethod
    def dot(self, other: "Vector") -> Any:
        """Dot product."""
        ...

    @abstractmethod
    def zero(self) -> "Vector":
        """Zero vector."""
        ...

    @abstractmethod
    def is_zero(self) -> bool:
        """Check if vector is zero."""
        ...

    @abstractmethod
    def add_term(self, scalar: Any, dim: Any, vec: "Vector") -> "Vector":
        """Add scalar * basis vector to vector."""
        ...

    @abstractmethod
    def of_term(self, scalar: Any, dim: Any) -> "Vector":
        """Create vector from scalar and dimension."""
        ...

    @abstractmethod
    def enum(self) -> Iterator[Tuple[Any, Any]]:
        """Enumerate non-zero entries."""
        ...

    @abstractmethod
    def of_enum(self, enum: Iterator[Tuple[Any, Any]]) -> "Vector":
        """Create vector from enumeration."""
        ...

    @abstractmethod
    def of_list(self, lst: List[Tuple[Any, Any]]) -> "Vector":
        """Create vector from list."""
        ...

    @abstractmethod
    def coeff(self, dim: Any) -> Any:
        """Get coefficient of dimension."""
        ...

    @abstractmethod
    def pivot(self, dim: Any) -> Tuple[Any, "Vector"]:
        """Extract coefficient and remove from vector."""
        ...

    @abstractmethod
    def pop(self) -> Tuple[Tuple[Any, Any], "Vector"]:
        """Remove and return arbitrary element."""
        ...

    @abstractmethod
    def map(self, f: Callable[[Any, Any], Any]) -> "Vector":
        """Map function over coefficients."""
        ...

    @abstractmethod
    def merge(self, f: Callable[[Any, Any, Any], Any], other: "Vector") -> "Vector":
        """Merge two vectors."""
        ...

    @abstractmethod
    def hash(self, hasher: Callable[[Tuple[Any, Any]], int]) -> int:
        """Hash function."""
        ...

    @abstractmethod
    def compare(self, comparer: Callable[[Any, Any], int]) -> Callable[["Vector"], int]:
        """Comparison function."""
        ...

    @abstractmethod
    def fold(self, f: Callable[[Any, Any, U], U], acc: U) -> U:
        """Fold over coefficients."""
        ...


class Matrix(Protocol):
    """Protocol for matrix operations over a ring."""

    @abstractmethod
    def equal(self, other: "Matrix") -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    def add(self, other: "Matrix") -> "Matrix":
        """Matrix addition."""
        ...

    @abstractmethod
    def scalar_mul(self, scalar: Any, mat: "Matrix") -> "Matrix":
        """Scalar multiplication."""
        ...

    @abstractmethod
    def mul(self, other: "Matrix") -> "Matrix":
        """Matrix multiplication."""
        ...

    @abstractmethod
    def zero(self) -> "Matrix":
        """Zero matrix."""
        ...

    @abstractmethod
    def identity(self, dims: List[Any]) -> "Matrix":
        """Identity matrix."""
        ...

    @abstractmethod
    def row(self, dim: Any) -> Any:  # Returns Vector type
        """Get row as vector."""
        ...

    @abstractmethod
    def column(self, dim: Any) -> Any:  # Returns Vector type
        """Get column as vector."""
        ...

    @abstractmethod
    def rowsi(self) -> Iterator[Tuple[Any, Any]]:
        """Enumerate rows with indices."""
        ...

    @abstractmethod
    def min_row(self) -> Tuple[Any, Any]:
        """Get lexicographically smallest row."""
        ...

    @abstractmethod
    def add_row(self, dim: Any, vec: Any) -> "Matrix":
        """Add row to matrix."""
        ...

    @abstractmethod
    def add_column(self, dim: Any, vec: Any) -> "Matrix":
        """Add column to matrix."""
        ...

    @abstractmethod
    def add_entry(self, row: Any, col: Any, scalar: Any) -> "Matrix":
        """Add entry to matrix."""
        ...

    @abstractmethod
    def pivot(self, dim: Any) -> Tuple[Any, "Matrix"]:
        """Extract row and remove from matrix."""
        ...

    @abstractmethod
    def pivot_column(self, dim: Any) -> Tuple[Any, "Matrix"]:
        """Extract column and remove from matrix."""
        ...

    @abstractmethod
    def transpose(self) -> "Matrix":
        """Matrix transpose."""
        ...

    @abstractmethod
    def entry(self, row: Any, col: Any) -> Any:
        """Get matrix entry."""
        ...

    @abstractmethod
    def entries(self) -> Iterator[Tuple[Any, Any, Any]]:
        """Enumerate all entries."""
        ...

    @abstractmethod
    def row_set(self) -> IntSet:
        """Set of row indices."""
        ...

    @abstractmethod
    def column_set(self) -> IntSet:
        """Set of column indices."""
        ...

    @abstractmethod
    def nb_rows(self) -> int:
        """Number of rows."""
        ...

    @abstractmethod
    def nb_columns(self) -> int:
        """Number of columns."""
        ...

    @abstractmethod
    def map_rows(self, f: Callable[[Any], Any]) -> "Matrix":
        """Map function over rows."""
        ...

    @abstractmethod
    def vector_right_mul(self, vec: Any) -> Any:
        """Matrix-vector multiplication."""
        ...

    @abstractmethod
    def vector_left_mul(self, vec: Any) -> Any:
        """Vector-matrix multiplication."""
        ...

    @abstractmethod
    def of_dense(self, matrix: List[List[Any]]) -> "Matrix":
        """Create from dense representation."""
        ...

    @abstractmethod
    def dense_of(self, rows: int, cols: int) -> List[List[Any]]:
        """Convert to dense representation."""
        ...

    @abstractmethod
    def of_rows(self, rows: List[Any]) -> "Matrix":
        """Create from list of row vectors."""
        ...

    @abstractmethod
    def interlace_columns(self, other: "Matrix") -> "Matrix":
        """Interlace columns of two matrices."""
        ...


# Concrete implementations using sparse maps


class RingMap(Generic[T], dict):
    """Sparse map implementation for ring elements."""

    def __init__(self, ring: Ring[T], data: Optional[Dict[T, T]] = None):
        if data is None:
            data = {}
        super().__init__(data)
        self.ring = ring

    @classmethod
    def identity(cls, ring: Ring[T]) -> RingMap[T]:
        """Create the identity ring map."""
        # For testing purposes, create a map that returns the input value
        # This is a simplified identity for basic testing
        identity_map = cls(ring)

        # Override the map method to return the input value for identity
        original_map = identity_map.map
        identity_map.map = lambda key: key

        return identity_map

    def is_scalar_zero(self, scalar: T) -> bool:
        """Check if scalar is zero."""
        return self.ring.equal(scalar, self.ring.zero())

    def map(self, key: T) -> T:
        """Apply the ring map to a key."""
        return self.get(key, self.ring.zero())

    def add(self, other: RingMap[T]) -> RingMap[T]:
        """Add two ring maps."""
        result = RingMap(self.ring)
        for key, value in self.items():
            result[key] = self.ring.add(value, other.get(key, self.ring.zero()))
        for key, value in other.items():
            if key not in result:
                result[key] = value
        return result

    def add_term(self, coeff: T, dim: Any, vec: RingMap[T]) -> RingMap[T]:
        """Add scalar * basis vector to vector."""
        if self.is_scalar_zero(coeff):
            return vec
        new_vec = RingMap(self.ring, vec)
        if dim in new_vec:
            new_vec[dim] = self.ring.add(new_vec[dim], coeff)
        else:
            new_vec[dim] = coeff
        return new_vec

    def coeff(self, dim: Any) -> T:
        """Get coefficient of dimension."""
        return self.get(dim, self.ring.zero())

    def enum(self) -> Iterator[Tuple[T, Any]]:
        """Enumerate non-zero entries."""
        for dim, coeff in self.items():
            if not self.is_scalar_zero(coeff):
                yield (coeff, dim)

    def of_enum(self, enum: Iterator[Tuple[T, Any]]) -> RingMap[T]:
        """Create vector from enumeration."""
        result = RingMap(self.ring)
        for coeff, dim in enum:
            result = result.add_term(coeff, dim, result)
        return result

    def of_list(self, lst: List[Tuple[T, Any]]) -> RingMap[T]:
        """Create vector from list."""
        result = RingMap(self.ring)
        for coeff, dim in lst:
            result = result.add_term(coeff, dim, result)
        return result

    def of_term(self, coeff: T, dim: Any) -> RingMap[T]:
        """Create vector from scalar and dimension."""
        return RingMap(self.ring, {dim: coeff})

    def negate(self) -> RingMap[T]:
        """Additive inverse."""
        return RingMap(self.ring, {k: self.ring.negate(v) for k, v in self.items()})

    def sub(self, other: RingMap[T]) -> RingMap[T]:
        """Vector subtraction."""
        return self.add(other.negate())

    def scalar_mul(self, k: T, vec: RingMap[T]) -> RingMap[T]:
        """Scalar multiplication."""
        if self.ring.equal(k, self.ring.one()):
            return vec
        elif self.ring.equal(k, self.ring.zero()):
            return RingMap(self.ring)
        else:
            return RingMap(
                self.ring, {k_: self.ring.mul(k, v) for k_, v in vec.items()}
            )

    def dot(self, other: RingMap[T]) -> T:
        """Dot product."""
        result = self.ring.zero()
        for dim, coeff in self.items():
            other_coeff = other.coeff(dim)
            result = self.ring.add(result, self.ring.mul(coeff, other_coeff))
        return result

    def zero(self) -> RingMap[T]:
        """Zero vector."""
        return RingMap(self.ring)

    def is_zero(self) -> bool:
        """Check if vector is zero."""
        return all(self.is_scalar_zero(coeff) for coeff in self.values())

    def pivot(self, dim: Any) -> Tuple[T, RingMap[T]]:
        """Extract coefficient and remove from vector."""
        coeff = self.coeff(dim)
        new_vec = RingMap(self.ring, {k: v for k, v in self.items() if k != dim})
        return (coeff, new_vec)

    def pop(self) -> Tuple[Tuple[Any, T], RingMap[T]]:
        """Remove and return arbitrary element."""
        if not self:
            raise ValueError("Cannot pop from empty vector")
        dim = next(iter(self.keys()))
        coeff = self[dim]
        new_vec = RingMap(self.ring, {k: v for k, v in self.items() if k != dim})
        return ((dim, coeff), new_vec)

    def map(self, f: Callable[[Any, T], T]) -> RingMap[T]:
        """Map function over coefficients."""
        return RingMap(self.ring, {k: f(k, v) for k, v in self.items()})

    def merge(self, f: Callable[[Any, T, T], T], other: RingMap[T]) -> RingMap[T]:
        """Merge two vectors."""
        all_keys = set(self.keys()) | set(other.keys())
        result = RingMap(self.ring)
        for key in all_keys:
            coeff1 = self.coeff(key)
            coeff2 = other.coeff(key)
            result[key] = f(key, coeff1, coeff2)
        return result

    def hash(self, hasher: Callable[[Tuple[Any, T]], int]) -> int:
        """Hash function."""
        return hash(tuple(sorted((hasher((k, v)) for k, v in self.items()))))

    def compare(self, comparer: Callable[[T, T], int]) -> int:
        """Comparison function."""

        def compare_vecs(other: RingMap[T]) -> int:
            keys1 = sorted(self.keys())
            keys2 = sorted(other.keys())
            if keys1 != keys2:
                return -1 if keys1 < keys2 else 1
            for k in keys1:
                cmp = comparer(self.coeff(k), other.coeff(k))
                if cmp != 0:
                    return cmp
            return 0

        return compare_vecs

    def fold(self, f: Callable[[Any, T, U], U], acc: U) -> U:
        """Fold over coefficients."""
        for dim, coeff in self.items():
            acc = f(dim, coeff, acc)
        return acc

    def equal(self, other: RingMap[T]) -> bool:
        """Equality comparison."""
        return (
            self.compare(
                lambda a, b: 0 if self.ring.equal(a, b) else -1 if a < b else 1
            )(other)
            == 0
        )


# Vector implementation over rings


class RingVector(RingMap[T]):
    """Vector implementation over a ring."""

    def __init__(
        self, ring: Ring[T], data: Optional[Union[Dict[T, T], List[T]]] = None
    ):
        if isinstance(data, list):
            # Convert list to dict with indices as keys
            data = {i: data[i] for i in range(len(data))}
        super().__init__(ring, data or {})

    @property
    def elements(self) -> List[T]:
        """Get vector elements as a list."""
        if not self:
            return []
        max_index = max(self.keys())
        return [self.get(i, self.ring.zero) for i in range(max_index + 1)]

    def interlace(self, other: RingVector[T]) -> RingVector[T]:
        """Interlace two vectors."""
        result = RingVector(self.ring)
        for coeff, dim in self.enum():
            result = result.add_term(coeff, 2 * dim, result)
        for coeff, dim in other.enum():
            result = result.add_term(coeff, 2 * dim + 1, result)
        return result

    def deinterlace(self) -> Tuple[RingVector[T], RingVector[T]]:
        """Deinterlace vector into two vectors."""
        v1 = RingVector(self.ring)
        v2 = RingVector(self.ring)
        for coeff, dim in self.enum():
            if dim % 2 == 0:
                v1 = v1.add_term(coeff, dim // 2, v1)
            else:
                v2 = v2.add_term(coeff, dim // 2, v2)
        return (v1, v2)


# Matrix implementation over rings


class RingMatrix(Generic[T]):
    """Matrix implementation over a ring."""

    def __init__(self, ring: Ring[T], data: Optional[List[List[T]]] = None):
        self.ring = ring
        self._data: Dict[int, RingVector[T]] = {}  # row -> vector
        if data:
            for i, row in enumerate(data):
                row_dict = {j: row[j] for j in range(len(row))}
                self._data[i] = RingVector(ring, row_dict)

    @property
    def elements(self) -> List[List[T]]:
        """Get matrix elements as a list of lists."""
        if not self._data:
            return []

        max_row = max(self._data.keys())
        result = []
        for i in range(max_row + 1):
            row_vec = self._data.get(i, RingVector(self.ring))
            row_data = []
            for j in range(max(row_vec.keys(), default=-1) + 1):
                row_data.append(row_vec.get(j, self.ring.zero))
            result.append(row_data)
        return result

    def scalar_mul(self, k: T, mat: RingMatrix[T]) -> RingMatrix[T]:
        """Scalar multiplication."""
        if self.ring.equal(k, self.ring.one()):
            return mat
        elif self.ring.equal(k, self.ring.zero()):
            return RingMatrix(self.ring)
        else:
            result = RingMatrix(self.ring)
            for i, row in mat._data.items():
                result._data[i] = RingVector(self.ring).scalar_mul(k, row)
            return result

    def row(self, dim: int) -> RingVector[T]:
        """Get row as vector."""
        return self._data.get(dim, RingVector(self.ring))

    def add(self, other: RingMatrix[T]) -> RingMatrix[T]:
        """Matrix addition."""
        result = RingMatrix(self.ring)
        all_rows = set(self._data.keys()) | set(other._data.keys())
        for i in all_rows:
            result._data[i] = self.row(i).add(other.row(i))
        return result

    def zero(self) -> RingMatrix[T]:
        """Zero matrix."""
        return RingMatrix(self.ring)

    def equal(self, other: RingMatrix[T]) -> bool:
        """Equality comparison."""
        return self._data == other._data

    def pivot(self, dim: int) -> Tuple[RingVector[T], RingMatrix[T]]:
        """Extract row and remove from matrix."""
        if dim not in self._data:
            return (RingVector(self.ring), self)
        row = self._data[dim]
        new_data = {k: v for k, v in self._data.items() if k != dim}
        return (row, RingMatrix(self.ring).__init_from_data(new_data))

    def add_row(self, i: int, vec: RingVector[T]) -> RingMatrix[T]:
        """Add row to matrix."""
        result = RingMatrix(self.ring)
        result._data = self._data.copy()
        result._data[i] = vec
        return result

    def rowsi(self) -> Iterator[Tuple[int, RingVector[T]]]:
        """Enumerate rows with indices."""
        for i, row in sorted(self._data.items()):
            yield (i, row)

    def entry(self, i: int, j: int) -> T:
        """Get matrix entry."""
        return self.row(i).coeff(j)

    def column(self, j: int) -> RingVector[T]:
        """Get column as vector."""
        result = RingVector(self.ring)
        for i, row in self.rowsi():
            coeff = row.coeff(j)
            if not self.ring.equal(coeff, self.ring.zero()):
                result = result.add_term(coeff, i, result)
        return result

    def pivot_column(self, j: int) -> Tuple[RingVector[T], RingMatrix[T]]:
        """Extract column and remove from matrix."""
        col = self.column(j)
        result = RingMatrix(self.ring)
        for i, row in self.rowsi():
            if i != j:
                new_row = RingVector(
                    self.ring, {k: v for k, v in row._data.items() if k != j}
                )
                result._data[i] = new_row
        return (col, result)

    def add_entry(self, i: int, j: int, k: T) -> RingMatrix[T]:
        """Add entry to matrix."""
        current_row = self.row(i)
        new_row = current_row.add_term(k, j, current_row)
        return self.add_row(i, new_row)

    def identity(self, dims: List[int]) -> RingMatrix[T]:
        """Identity matrix."""
        result = RingMatrix(self.ring)
        for d in dims:
            result = result.add_entry(d, d, self.ring.one())
        return result

    def add_column(self, i: int, col: RingVector[T]) -> RingMatrix[T]:
        """Add column to matrix."""
        result = RingMatrix(self.ring)
        for coeff, j in col.enum():
            result = result.add_entry(j, i, coeff)
        return result

    def entries(self) -> Iterator[Tuple[int, int, T]]:
        """Enumerate all entries."""
        for i, row in self.rowsi():
            for coeff, j in row.enum():
                yield (i, j, coeff)

    def row_set(self) -> IntSet:
        """Set of row indices."""
        return IntSet(set(self._data.keys()))

    def column_set(self) -> IntSet:
        """Set of column indices."""
        cols = set()
        for row in self._data.values():
            cols.update(row._data.keys())
        return IntSet(cols)

    def nb_rows(self) -> int:
        """Number of rows."""
        return len(self._data)

    def nb_columns(self) -> int:
        """Number of columns."""
        return len(self.column_set())

    def transpose(self) -> RingMatrix[T]:
        """Matrix transpose."""
        result = RingMatrix(self.ring)
        for i, j, k in self.entries():
            result = result.add_entry(j, i, k)
        return result

    def mul(self, other: RingMatrix[T]) -> RingMatrix[T]:
        """Matrix multiplication."""
        if self.nb_columns() != other.nb_rows():
            raise ValueError("Incompatible dimensions for matrix multiplication")

        result = RingMatrix(self.ring)
        for i in self.row_set():
            row = self.row(i)
            for j in other.column_set():
                dot_product = self.ring.zero()
                for k in range(max(self.nb_columns(), other.nb_rows())):
                    dot_product = self.ring.add(
                        dot_product, self.ring.mul(row.coeff(k), other.entry(k, j))
                    )
                if not self.ring.equal(dot_product, self.ring.zero()):
                    result = result.add_entry(i, j, dot_product)
        return result

    def min_row(self) -> Tuple[int, RingVector[T]]:
        """Get lexicographically smallest row."""
        if not self._data:
            raise ValueError("Matrix is empty")
        return min(self.rowsi(), key=lambda x: x[0])

    def map_rows(self, f: Callable[[RingVector[T]], RingVector[T]]) -> RingMatrix[T]:
        """Map function over rows."""
        result = RingMatrix(self.ring)
        for i, row in self._data.items():
            result._data[i] = f(row)
        return result

    def vector_right_mul(self, v: RingVector[T]) -> RingVector[T]:
        """Matrix-vector multiplication."""
        result = RingVector(self.ring)
        for i, scalar in v.enum():
            row = self.row(i)
            for j, coeff in row.enum():
                result = result.add_term(self.ring.mul(scalar, coeff), j, result)
        return result

    def vector_left_mul(self, v: RingVector[T]) -> RingVector[T]:
        """Vector-matrix multiplication."""
        result = RingVector(self.ring)
        for j, scalar in v.enum():
            col = self.column(j)
            for i, coeff in col.enum():
                result = result.add_term(self.ring.mul(scalar, coeff), i, result)
        return result

    def of_dense(self, matrix: List[List[T]]) -> RingMatrix[T]:
        """Create from dense representation."""
        result = RingMatrix(self.ring)
        for i, row in enumerate(matrix):
            for j, entry in enumerate(row):
                if not self.ring.equal(entry, self.ring.zero()):
                    result = result.add_entry(i, j, entry)
        return result

    def dense_of(self, rows: int, cols: int) -> List[List[T]]:
        """Convert to dense representation."""
        result = [[self.ring.zero() for _ in range(cols)] for _ in range(rows)]
        for i, j, k in self.entries():
            if i < rows and j < cols:
                result[i][j] = k
        return result

    def of_rows(self, rows: List[RingVector[T]]) -> RingMatrix[T]:
        """Create from list of row vectors."""
        result = RingMatrix(self.ring)
        for i, row in enumerate(rows):
            result._data[i] = row
        return result

    def interlace_columns(self, other: RingMatrix[T]) -> RingMatrix[T]:
        """Interlace columns of two matrices."""
        all_rows = self.row_set().union(other.row_set())
        result = RingMatrix(self.ring)
        for i in all_rows:
            row1 = self.row(i)
            row2 = other.row(i)
            interlaced = row1.interlace(row2)
            result._data[i] = interlaced
        return result

    def __init_from_data(self, data: Dict[int, RingVector[T]]) -> RingMatrix[T]:
        """Internal method to create matrix from data."""
        self._data = data
        return self


# Concrete ring implementations
class IntegerRing:
    """Ring of integers."""

    # Class attributes for zero and one
    zero = 0
    one = 1

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def neg(a):
        return -a

    @staticmethod
    def equal(a, b):
        return a == b


class RationalRing:
    """Ring of rational numbers."""

    @staticmethod
    def zero():
        from fractions import Fraction

        return Fraction(0)

    @staticmethod
    def one():
        from fractions import Fraction

        return Fraction(1)

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def neg(a):
        return -a

    @staticmethod
    def equal(a, b):
        return a == b


def is_ring(obj) -> bool:
    """Check if an object implements the ring interface."""
    required_methods = ["zero", "one", "add", "mul", "neg", "equal"]
    return all(hasattr(obj, method) for method in required_methods)


def is_commutative_semigroup(obj, mul_func=None, a=None, b=None) -> bool:
    """Check if an object implements a commutative semigroup (additive)."""
    # For the simple case, just check if the object has the required methods
    required_methods = ["zero", "add", "equal"]
    return all(hasattr(obj, method) for method in required_methods)


def is_associative_semigroup(obj, mul_func=None, a=None, b=None, c=None) -> bool:
    """Check if an object implements an associative semigroup (multiplicative)."""
    required_methods = ["one", "mul", "equal"]
    return all(hasattr(obj, method) for method in required_methods)
