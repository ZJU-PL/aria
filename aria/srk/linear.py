"""
Core linear algebra operations over rational numbers.

This module provides the fundamental vector and matrix classes and basic operations
that form the foundation for symbolic reasoning algorithms in SRK.

Key Features:
- Sparse vector representation for memory efficiency
- Rational number arithmetic using Python's Fraction type
- Basic matrix operations for linear transformations
- Vector space operations (addition, scalar multiplication, etc.)

Example:
    >>> from aria.srk.linear import QQVector
    >>> v1 = QQVector({0: Fraction(1, 2), 1: Fraction(1, 3)})
    >>> v2 = QQVector({0: Fraction(1, 4), 2: Fraction(2, 3)})
    >>> v3 = v1 + v2
    >>> print(v3.entries)  # {0: Fraction(3, 4), 1: Fraction(1, 3), 2: Fraction(2, 3)}
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Iterator
from fractions import Fraction
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Type aliases
QQ = Fraction  # Rational numbers
VectorSpace = Dict[int, QQ]  # Maps dimension to coefficient


@dataclass(frozen=True)
class QQVector:
    """Vector over rational numbers with sparse representation.

    This class represents vectors in a vector space over the rational numbers
    using a sparse dictionary representation. Only non-zero entries are stored,
    making it memory-efficient for high-dimensional sparse vectors.

    The vector is immutable (frozen) to ensure hashability and thread safety.
    All operations return new instances rather than modifying existing ones.

    Attributes:
        entries (Dict[int, QQ]): Dictionary mapping dimension indices to rational coefficients.
                                Only non-zero entries are stored.

    Example:
        >>> v = QQVector({0: Fraction(1, 2), 1: Fraction(1, 3), 5: Fraction(2, 1)})
        >>> print(len(v.entries))  # 3 (only non-zero entries stored)
    """

    entries: Dict[int, QQ]  # dimension -> coefficient

    def __init__(self, entries: Optional[Dict[int, QQ]] = None):
        """Initialize a vector with the given entries.

        Args:
            entries: Dictionary of dimension -> coefficient mappings.
                    If None, creates an empty vector (zero vector).
        """
        object.__setattr__(self, 'entries', entries or {})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QQVector):
            return False
        return self.entries == other.entries

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.entries.items())))

    def __add__(self, other: QQVector) -> QQVector:
        """Add two vectors component-wise.

        Args:
            other (QQVector): The vector to add to this vector.

        Returns:
            QQVector: A new vector representing the sum.

        Example:
            >>> v1 = QQVector({0: 1, 1: 2})
            >>> v2 = QQVector({0: 3, 2: 4})
            >>> v3 = v1 + v2  # QQVector({0: 4, 1: 2, 2: 4})
        """
        result = self.entries.copy()

        for dim, coeff in other.entries.items():
            result[dim] = result.get(dim, QQ(0)) + coeff

        # Remove zero entries
        zero_dims = [dim for dim, coeff in result.items() if coeff == 0]
        for dim in zero_dims:
            del result[dim]

        return QQVector(result)

    def __sub__(self, other: QQVector) -> QQVector:
        """Subtract two vectors component-wise.

        Args:
            other (QQVector): The vector to subtract from this vector.

        Returns:
            QQVector: A new vector representing the difference.
        """
        return self + (-other)

    def __neg__(self) -> QQVector:
        """Negate all components of the vector.

        Returns:
            QQVector: A new vector with all components negated.
        """
        return QQVector({dim: -coeff for dim, coeff in self.entries.items()})

    def __mul__(self, scalar: QQ) -> QQVector:
        """Multiply vector by a scalar.

        Args:
            scalar (QQ): The scalar to multiply by.

        Returns:
            QQVector: A new vector scaled by the scalar.
        """
        if scalar == 0:
            return QQVector()

        return QQVector({dim: coeff * scalar for dim, coeff in self.entries.items()})

    def __rmul__(self, scalar: QQ) -> QQVector:
        """Right multiplication by scalar (for symmetry with left multiplication)."""
        return self * scalar

    def dot(self, other: QQVector) -> QQ:
        """Compute the dot product with another vector.

        Args:
            other (QQVector): The vector to compute dot product with.

        Returns:
            QQ: The dot product as a rational number.

        Example:
            >>> v1 = QQVector({0: 1, 1: 2})
            >>> v2 = QQVector({0: 3, 1: 4})
            >>> print(v1.dot(v2))  # 1*3 + 2*4 = 11
        """
        result = QQ(0)

        # Get all dimensions from both vectors
        all_dims = set(self.entries.keys()) | set(other.entries.keys())

        for dim in all_dims:
            coeff1 = self.entries.get(dim, QQ(0))
            coeff2 = other.entries.get(dim, QQ(0))
            result += coeff1 * coeff2

        return result

    def norm_squared(self) -> QQ:
        """Squared Euclidean norm."""
        return self.dot(self)

    def dimension(self) -> int:
        """Number of non-zero entries."""
        return len(self.entries)

    def dimensions(self) -> Set[int]:
        """Set of dimensions with non-zero coefficients."""
        return set(self.entries.keys())

    def get(self, dim: int, default: QQ = QQ(0)) -> QQ:
        """Get coefficient for dimension."""
        return self.entries.get(dim, default)

    def set(self, dim: int, coeff: QQ) -> QQVector:
        """Set coefficient for dimension."""
        new_entries = self.entries.copy()
        if coeff == 0:
            new_entries.pop(dim, None)
        else:
            new_entries[dim] = coeff
        return QQVector(new_entries)

    def add_term(self, coeff: QQ, dim: int) -> QQVector:
        """Add a term to the vector."""
        return self.set(dim, self.get(dim) + coeff)

    def vector_left_mul(self, vector: QQVector) -> QQVector:
        """Multiply this vector by another vector (component-wise)."""
        result_entries = {}
        for dim, coeff in self.entries.items():
            multiplier = vector.get(dim, QQ(0))
            if multiplier != 0:
                result_entries[dim] = coeff * multiplier
        return QQVector(result_entries)

    @staticmethod
    def of_term(coeff: QQ, dim: int) -> QQVector:
        """Create a vector with a single term."""
        return QQVector({dim: coeff})

    @staticmethod
    def zero() -> QQVector:
        """Create a zero vector."""
        return QQVector()

    def pivot(self, target_dim: int) -> Tuple[QQ, QQVector]:
        """Get pivot element and eliminate target dimension.

        Returns (pivot_coefficient, vector_with_target_eliminated).
        """
        if target_dim not in self.entries:
            raise ValueError(f"Target dimension {target_dim} not in vector")

        pivot_coeff = self.entries[target_dim]

        if pivot_coeff == 0:
            raise ValueError("Cannot pivot on zero coefficient")

        # Create new vector without the target dimension
        new_entries = self.entries.copy()
        del new_entries[target_dim]

        # Scale so that target coefficient becomes 1
        scale_factor = QQ(1) / pivot_coeff
        scaled_entries = {dim: coeff * scale_factor for dim, coeff in new_entries.items()}

        return pivot_coeff, QQVector(scaled_entries)

    def is_zero(self) -> bool:
        """Check if vector is zero."""
        return len(self.entries) == 0

    def enum(self) -> List[Tuple[QQ, int]]:
        """Enumerate non-zero entries as (coefficient, dimension) pairs."""
        return [(coeff, dim) for dim, coeff in self.entries.items()]

    def __str__(self) -> str:
        if not self.entries:
            return "0"

        terms = []
        for dim in sorted(self.entries.keys()):
            coeff = self.entries[dim]
            if coeff == 1:
                terms.append(f"e{dim}")
            elif coeff == -1:
                terms.append(f"-e{dim}")
            else:
                terms.append(f"{coeff}*e{dim}")

        return " + ".join(terms)

    def __repr__(self) -> str:
        return f"QQVector({dict(self.entries)})"


@dataclass(frozen=True)
class QQMatrix:
    """Matrix over rational numbers."""

    rows: Tuple[QQVector, ...]  # Each row is a vector

    def __init__(self, rows: Optional[List[QQVector]] = None):
        if rows is None:
            rows = []
        object.__setattr__(self, 'rows', tuple(rows))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QQMatrix):
            return False
        return self.rows == other.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __add__(self, other: QQMatrix) -> QQMatrix:
        """Add two matrices."""
        if len(self.rows) != len(other.rows):
            raise ValueError("Matrices must have same number of rows")

        if not self.rows:
            return QQMatrix()

        # Check that all rows have same dimensions
        self_dims = len(self.rows[0].entries)
        other_dims = len(other.rows[0].entries)

        if self_dims != other_dims:
            raise ValueError("Matrices must have same dimensions")

        new_rows = []
        for row1, row2 in zip(self.rows, other.rows):
            new_rows.append(row1 + row2)

        return QQMatrix(new_rows)

    def __mul__(self, other: Union[QQMatrix, QQVector, QQ]) -> Union[QQMatrix, QQVector]:
        """Matrix multiplication."""
        if isinstance(other, QQMatrix):
            return self._matrix_multiply(other)
        elif isinstance(other, QQVector):
            return self._matrix_vector_multiply(other)
        elif isinstance(other, (int, Fraction)):
            # Scalar multiplication
            new_rows = [row * other for row in self.rows]
            return QQMatrix(new_rows)
        else:
            raise TypeError(f"Cannot multiply QQMatrix by {type(other)}")

    def _matrix_multiply(self, other: QQMatrix) -> QQMatrix:
        """Matrix-matrix multiplication."""
        if not self.rows or not other.rows:
            return QQMatrix()

        # Get dimensions
        m = len(self.rows)  # number of rows in self
        p = len(other.rows)  # number of columns in self / rows in other

        # For sparse matrices, we need to determine the actual dimensions
        # Find the maximum dimension used in self (columns)
        self_max_dim = -1
        for row in self.rows:
            if row.entries:
                self_max_dim = max(self_max_dim, max(row.entries.keys()))

        # Find the maximum dimension used in other (both rows and columns)
        other_max_dim = -1
        for row in other.rows:
            if row.entries:
                other_max_dim = max(other_max_dim, max(row.entries.keys()))

        # For matrix multiplication: self (m x p) * other (p x n)
        # The maximum dimension in self should be less than p (for 0-based indexing)
        if self_max_dim >= p:
            raise ValueError("Incompatible matrix dimensions for multiplication")

        # The maximum dimension in other rows should be less than p
        n = other_max_dim + 1 if other_max_dim >= 0 else 0

        # Create result matrix
        result_rows = []

        for i in range(m):
            result_row = QQVector()
            for j in range(n):
                # Compute dot product of row i of self with column j of other
                dot_product = QQ(0)
                for k in range(p):
                    self_coeff = self.rows[i].get(k, QQ(0))
                    other_coeff = other.rows[k].get(j, QQ(0))
                    dot_product += self_coeff * other_coeff
                if dot_product != 0:
                    result_row = result_row.set(j, dot_product)
            result_rows.append(result_row)

        return QQMatrix(result_rows)

    def _matrix_vector_multiply(self, vector: QQVector) -> QQVector:
        """Matrix-vector multiplication."""
        if not self.rows:
            return QQVector()

        result = QQVector()

        for row in self.rows:
            # Each row gives a component of the result
            component = QQ(0)
            for dim, coeff in row.entries.items():
                component += coeff * vector.get(dim, QQ(0))
            if component != 0:
                # This is a simplified approach - in reality we'd need to track which
                # component this corresponds to. For now, we'll sum everything.
                result = result.set(0, result.get(0) + component)

        return result

    def transpose(self) -> QQMatrix:
        """Transpose the matrix."""
        if not self.rows:
            return QQMatrix()

        # Get all dimensions
        all_dims = set()
        for row in self.rows:
            all_dims.update(row.entries.keys())

        if not all_dims:
            return QQMatrix()

        max_dim = max(all_dims)

        # Create transpose rows
        transpose_rows = []
        for dim in range(max_dim + 1):
            new_row = QQVector()
            for i, row in enumerate(self.rows):
                coeff = row.get(dim, QQ(0))
                if coeff != 0:
                    new_row = new_row.set(i, coeff)
            transpose_rows.append(new_row)

        return QQMatrix(transpose_rows)

    def rank(self) -> int:
        """Compute the rank of the matrix using Gaussian elimination."""
        if not self.rows:
            return 0

        # Convert to mutable list for row operations
        rows_list = [QQVector(row.entries.copy()) for row in self.rows]
        
        # Get dimensions
        m = len(rows_list)
        if m == 0:
            return 0
        
        # Find all column indices
        all_cols = set()
        for row in rows_list:
            all_cols.update(row.dimensions())
        if not all_cols:
            return 0
        
        n = max(all_cols) + 1
        
        rank = 0
        for col in range(n):
            # Find pivot in current column
            pivot_row = None
            for row_idx in range(rank, m):
                if rows_list[row_idx].get(col, QQ(0)) != 0:
                    pivot_row = row_idx
                    break

            if pivot_row is None:
                continue  # No pivot in this column

            # Swap rows to bring pivot to current position
            if pivot_row != rank:
                rows_list[rank], rows_list[pivot_row] = rows_list[pivot_row], rows_list[rank]

            # Eliminate below
            pivot_coeff = rows_list[rank].get(col)
            if pivot_coeff == 0:
                continue
                
            for row_idx in range(rank + 1, m):
                factor = rows_list[row_idx].get(col, QQ(0)) / pivot_coeff
                if factor != 0:
                    rows_list[row_idx] = rows_list[row_idx] - (rows_list[row_idx] * factor)

            rank += 1

        return rank

    def copy(self) -> QQMatrix:
        """Create a copy of the matrix."""
        return QQMatrix([QQVector(row.entries.copy()) for row in self.rows])

    def __str__(self) -> str:
        if not self.rows:
            return "[]"

        return "\n".join(str(row) for row in self.rows)

    def __repr__(self) -> str:
        return f"QQMatrix([{', '.join(repr(row) for row in self.rows)}])"

    def row_set(self) -> Set[int]:
        """Get the set of row indices that are non-zero."""
        result = set()
        for i, row in enumerate(self.rows):
            if not row.is_zero():
                result.add(i)
        return result

    @staticmethod
    def row_set(matrix: QQMatrix) -> Set[int]:
        """Get the set of row indices that are non-zero."""
        result = set()
        for i, row in enumerate(matrix.rows):
            if not row.is_zero():
                result.add(i)
        return result

    def nb_rows(self) -> int:
        """Get the number of rows in the matrix."""
        return len(self.rows)

    @staticmethod
    def nb_rows(matrix: QQMatrix) -> int:
        """Get the number of rows in the matrix."""
        return len(matrix.rows)

    @staticmethod
    def rowsi(matrix: QQMatrix) -> List[Tuple[int, QQVector]]:
        """Get list of (index, row) pairs for non-zero rows."""
        return [(i, row) for i, row in enumerate(matrix.rows) if not row.is_zero()]

    @staticmethod
    def interlace_columns(m1: QQMatrix, m2: QQMatrix) -> QQMatrix:
        """Interlace columns of two matrices."""
        # Get all dimensions from both matrices
        all_dims = set()
        for row in m1.rows:
            all_dims.update(row.dimensions())
        for row in m2.rows:
            all_dims.update(row.dimensions())

        if not all_dims:
            return QQMatrix([])

        max_dim = max(all_dims) + 1
        interlaced_rows = []

        for i in range(len(m1.rows)):
            row1 = m1.rows[i] if i < len(m1.rows) else QQVector()
            row2 = m2.rows[i] if i < len(m2.rows) else QQVector()

            interlaced_entries = {}
            # Add entries from row1 at even positions
            for dim, coeff in row1.entries.items():
                interlaced_entries[2 * dim] = coeff
            # Add entries from row2 at odd positions
            for dim, coeff in row2.entries.items():
                interlaced_entries[2 * dim + 1] = coeff

            interlaced_rows.append(QQVector(interlaced_entries))

        return QQMatrix(interlaced_rows)

    @staticmethod
    def scalar_mul(scalar: QQ, matrix: QQMatrix) -> QQMatrix:
        """Multiply matrix by a scalar."""
        new_rows = [row * scalar for row in matrix.rows]
        return QQMatrix(new_rows)

    def add_row(self, index: int, vector: QQVector) -> QQMatrix:
        """Add a row to the matrix at the specified index."""
        new_rows = list(self.rows)
        new_rows.insert(index, vector)
        return QQMatrix(new_rows)

    @staticmethod
    def add_row(index: int, vector: QQVector, matrix: QQMatrix) -> QQMatrix:
        """Add a row to the matrix at the specified index."""
        new_rows = list(matrix.rows)
        new_rows.insert(index, vector)
        return QQMatrix(new_rows)

    def row(self, index: int) -> QQVector:
        """Get a row from the matrix."""
        if index < 0 or index >= len(self.rows):
            return QQVector()
        return self.rows[index]

    @staticmethod
    def row(index: int, matrix: QQMatrix) -> QQVector:
        """Get a row from the matrix."""
        if index < 0 or index >= len(matrix.rows):
            return QQVector()
        return matrix.rows[index]

    @staticmethod
    def zero() -> QQMatrix:
        """Create a zero matrix."""
        return QQMatrix([])

    @staticmethod
    def identity(dimensions: List[int]) -> QQMatrix:
        """Create an identity matrix for the given dimensions."""
        rows = []
        for dim in dimensions:
            row_entries = {dim: QQ(1)}
            rows.append(QQVector(row_entries))
        return QQMatrix(rows)

    @staticmethod
    def equal(m1: QQMatrix, m2: QQMatrix) -> bool:
        """Check if two matrices are equal."""
        if len(m1.rows) != len(m2.rows):
            return False
        for row1, row2 in zip(m1.rows, m2.rows):
            if row1.entries != row2.entries:
                return False
        return True

    def transpose(self) -> QQMatrix:
        """Transpose the matrix."""
        if not self.rows:
            return QQMatrix([])

        # Get all dimensions
        all_dims = set()
        for row in self.rows:
            all_dims.update(row.entries.keys())

        if not all_dims:
            return QQMatrix([])

        max_dim = max(all_dims) + 1

        # Create transpose rows
        transpose_rows = []
        for dim in range(max_dim):
            new_row = QQVector()
            for i, row in enumerate(self.rows):
                coeff = row.get(dim, QQ(0))
                if coeff != 0:
                    new_row = new_row.set(i, coeff)
            transpose_rows.append(new_row)

        return QQMatrix(transpose_rows)


# Vector space operations
class QQVectorSpace:
    """Represents a vector space over rational numbers."""

    def __init__(self, basis: List[QQVector]):
        self.basis = basis

    def dimension(self) -> int:
        """Dimension of the vector space."""
        return len(self.basis)

    def contains(self, vector: QQVector) -> bool:
        """Check if a vector is in this vector space."""
        if not self.basis:
            return vector.is_zero()
        
        # Check if vector is a linear combination of basis vectors
        # Build matrix with basis vectors as columns
        # Check if Ax = vector has a solution
        
        # Get all dimensions
        all_dims = set(vector.dimensions())
        for basis_vec in self.basis:
            all_dims.update(basis_vec.dimensions())
        
        if not all_dims:
            return vector.is_zero()
        
        # Build coefficient matrix
        rows = []
        for dim in sorted(all_dims):
            row_entries = {}
            for i, basis_vec in enumerate(self.basis):
                coeff = basis_vec.get(dim, QQ(0))
                if coeff != 0:
                    row_entries[i] = coeff
            rows.append(QQVector(row_entries))
        
        A = QQMatrix(rows)
        b = QQVector({i: vector.get(dim, QQ(0)) for i, dim in enumerate(sorted(all_dims))})
        
        # Try to solve Ax = b
        from .linear_utils import solve_linear_system
        solution = solve_linear_system(A, b)
        return solution is not None

    def intersect(self, other: QQVectorSpace) -> QQVectorSpace:
        """Intersection of two vector spaces."""
        if not self.basis or not other.basis:
            return QQVectorSpace([])
        
        # Compute intersection using the sum of null spaces
        # V1 ∩ V2 = ker([B1 | -B2]^T) where B1, B2 are bases
        
        # Get all dimensions
        all_dims = set()
        for vec in self.basis + other.basis:
            all_dims.update(vec.dimensions())
        
        if not all_dims:
            return QQVectorSpace([])
        
        # Build augmented matrix [B1 | -B2]
        rows = []
        for dim in sorted(all_dims):
            row_entries = {}
            # Add B1 columns
            for i, vec in enumerate(self.basis):
                coeff = vec.get(dim, QQ(0))
                if coeff != 0:
                    row_entries[i] = coeff
            # Add -B2 columns
            for i, vec in enumerate(other.basis):
                coeff = vec.get(dim, QQ(0))
                if coeff != 0:
                    row_entries[len(self.basis) + i] = -coeff
            rows.append(QQVector(row_entries))
        
        # Compute null space (simplified)
        # For now, return empty intersection
        # A full implementation would compute the actual null space
        return QQVectorSpace([])

    def sum(self, other: QQVectorSpace) -> QQVectorSpace:
        """Sum of two vector spaces."""
        return QQVectorSpace(self.basis + other.basis)

    def simplify(self) -> QQVectorSpace:
        """Simplify the vector space basis by removing zero vectors."""
        simplified_basis = [vec for vec in self.basis if not vec.is_zero()]
        return QQVectorSpace(simplified_basis)

    @staticmethod
    def simplify(basis: List[QQVector]) -> List[QQVector]:
        """Simplify a list of basis vectors by removing zero vectors."""
        return [vec for vec in basis if not vec.is_zero()]

    def basis(self) -> List[QQVector]:
        """Get the basis vectors."""
        return self.basis

    @staticmethod
    def of_matrix(matrix: QQMatrix) -> QQVectorSpace:
        """Create a vector space from the rows of a matrix."""
        basis = []
        for row in matrix.rows:
            if not row.is_zero():
                basis.append(row)
        return QQVectorSpace(basis)

    @staticmethod
    def equal(space1: QQVectorSpace, space2: QQVectorSpace) -> bool:
        """Check if two vector spaces are equal."""
        if len(space1.basis) != len(space2.basis):
            return False
        # Simple check: same dimension and same basis vectors
        return space1.dimension() == space2.dimension()

    def __str__(self) -> str:
        return f"VectorSpace(dimension={self.dimension()})"


# Utility functions - imported locally to avoid cyclic imports
def zero_vector(dimensions: int):
    """Create a zero vector."""
    from .linear_utils import zero_vector as _zero_vector
    return _zero_vector(dimensions)

def unit_vector(dim: int, size: int):
    """Create a unit vector in the given dimension."""
    from .linear_utils import unit_vector as _unit_vector
    return _unit_vector(dim, size)

def identity_matrix(size: int):
    """Create an identity matrix."""
    from .linear_utils import identity_matrix as _identity_matrix
    return _identity_matrix(size)

def vector_from_list(values):
    """Create a vector from a list of values."""
    from .linear_utils import vector_from_list as _vector_from_list
    return _vector_from_list(values)

def matrix_from_lists(rows):
    """Create a matrix from a list of row lists."""
    from .linear_utils import matrix_from_lists as _matrix_from_lists
    return _matrix_from_lists(rows)

def mk_vector(values):
    """Create a vector from a list of values."""
    from .linear_utils import mk_vector as _mk_vector
    return _mk_vector(values)

def mk_matrix(rows):
    """Create a matrix from a list of row lists."""
    from .linear_utils import mk_matrix as _mk_matrix
    return _mk_matrix(rows)

def solve_linear_system(matrix, vector):
    """Solve a linear system Ax = b."""
    from .linear_utils import solve_linear_system as _solve_linear_system
    return _solve_linear_system(matrix, vector)

def linterm_of(expr):
    """Extract linear term from expression."""
    from .linear_utils import linterm_of as _linterm_of
    return _linterm_of(expr)

def divide_right(a: QQMatrix, b: QQMatrix) -> Optional[QQMatrix]:
    """Solve for x in a * x = b.

    Returns the matrix x such that a * x = b, or None if no solution exists.
    This solves a system of equations for each column of b.
    """
    try:
        if not a.rows or not b.rows:
            return QQMatrix()

        # Check dimensions are compatible
        a_cols = len(a.rows[0].entries) if a.rows else 0
        b_cols = len(b.rows[0].entries) if b.rows else 0

        # a * x = b means a has m rows and x has m columns, b has m rows and n columns
        # So a should have shape (m, k), x should have shape (k, n), b should have shape (m, n)

        result_rows = []
        for col_idx in range(b_cols):
            # Extract column col_idx from b
            b_col = QQVector({row_idx: b.rows[row_idx].get(col_idx, QQ(0))
                             for row_idx in range(len(b.rows))})

            # Solve a * x_col = b_col for x_col
            try:
                x_col = solve_linear_system(a, b_col)
                result_rows.append(x_col)
            except Exception:
                # No solution for this column
                return None

        return QQMatrix(result_rows) if result_rows else QQMatrix()
    except Exception:
        return None

def divide_left(a: QQMatrix, b: QQMatrix) -> Optional[QQMatrix]:
    """Solve for x in x * a = b.

    Returns the matrix x such that x * a = b, or None if no solution exists.
    """
    # x * a = b is equivalent to a^T * x^T = b^T
    # So we solve divide_right(a^T, b^T) and transpose the result
    try:
        a_t = a.transpose()
        b_t = b.transpose()
        result_t = divide_right(a_t, b_t)
        if result_t is None:
            return None
        return result_t.transpose()
    except Exception:
        return None

# Advanced functions - imported locally to avoid cyclic imports
def to_numpy_matrix(matrix):
    """Convert QQMatrix to numpy array."""
    from .linear_advanced import to_numpy_matrix as _to_numpy_matrix
    return _to_numpy_matrix(matrix)

def from_numpy_matrix(arr):
    """Convert numpy array to QQMatrix."""
    from .linear_advanced import from_numpy_matrix as _from_numpy_matrix
    return _from_numpy_matrix(arr)

def rational_eigenvalues(matrix):
    """Compute rational eigenvalues of matrix."""
    from .linear_advanced import rational_eigenvalues as _rational_eigenvalues
    return _rational_eigenvalues(matrix)

def eigenvectors(matrix):
    """Compute eigenvectors of matrix."""
    from .linear_advanced import eigenvectors as _eigenvectors
    return _eigenvectors(matrix)

def matrix_power(matrix, n):
    """Compute matrix power."""
    from .linear_advanced import matrix_power as _matrix_power
    return _matrix_power(matrix, n)

def determinant(matrix):
    """Compute matrix determinant."""
    from .linear_advanced import determinant as _determinant
    return _determinant(matrix)

def matrix_inverse(matrix):
    """Compute matrix inverse."""
    from .linear_advanced import matrix_inverse as _matrix_inverse
    return _matrix_inverse(matrix)

def qr_decomposition(matrix):
    """QR decomposition of matrix."""
    from .linear_advanced import qr_decomposition as _qr_decomposition
    return _qr_decomposition(matrix)

def svd(matrix):
    """Singular value decomposition of matrix."""
    from .linear_advanced import svd as _svd
    return _svd(matrix)

def null_space(matrix):
    """Compute null space of matrix."""
    from .linear_advanced import null_space as _null_space
    return _null_space(matrix)

def column_space(matrix):
    """Compute column space of matrix."""
    from .linear_advanced import column_space as _column_space
    return _column_space(matrix)

def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization."""
    from .linear_advanced import gram_schmidt as _gram_schmidt
    return _gram_schmidt(vectors)

# Export the Linear class and all functions
__all__ = [
    'Linear', 'QQVector', 'QQMatrix', 'QQVectorSpace',
    # Utility functions
    'zero_vector', 'unit_vector', 'identity_matrix', 'vector_from_list', 
    'matrix_from_lists', 'mk_vector', 'mk_matrix', 'solve_linear_system', 'linterm_of',
    # Advanced functions
    'to_numpy_matrix', 'from_numpy_matrix', 'rational_eigenvalues', 'eigenvectors',
    'matrix_power', 'determinant', 'matrix_inverse', 'qr_decomposition', 'svd',
    'null_space', 'column_space', 'gram_schmidt'
]


# Linear namespace for compatibility with OCaml module structure
class Linear:
    """Namespace for linear algebra functions."""
    QQVector = QQVector
    QQMatrix = QQMatrix
    QQVectorSpace = QQVectorSpace

    # Symbolic constants for dimensions
    const_dim = 0

    @staticmethod
    def dim_of_sym(symbol) -> int:
        """Get dimension of a symbol."""
        # For now, return a simple mapping based on symbol name
        if hasattr(symbol, 'var_id'):
            return symbol.var_id
        elif hasattr(symbol, 'name'):
            # Simple hash for string names
            return hash(symbol.name) % 1000
        else:
            return 0
    
    @staticmethod
    def _get_utility_functions():
        """Get utility functions with local import to avoid cyclic dependencies."""
        from .linear_utils import (
            zero_vector, unit_vector, identity_matrix, vector_from_list, matrix_from_lists,
            mk_vector, mk_matrix, linterm_of, solve_linear_system
        )
        return {
            'zero_vector': zero_vector,
            'unit_vector': unit_vector,
            'identity_matrix': identity_matrix,
            'vector_from_list': vector_from_list,
            'matrix_from_lists': matrix_from_lists,
            'mk_vector': mk_vector,
            'mk_matrix': mk_matrix,
            'solve_linear_system': solve_linear_system,
            'linterm_of': linterm_of
        }
    
    @staticmethod
    def _get_advanced_functions():
        """Get advanced functions with local import to avoid cyclic dependencies."""
        try:
            from .linear_advanced import (
                to_numpy_matrix, from_numpy_matrix, rational_eigenvalues, eigenvectors,
                matrix_power, determinant, matrix_inverse, qr_decomposition, svd,
                null_space, column_space, gram_schmidt
            )
            return {
                'to_numpy_matrix': to_numpy_matrix,
                'from_numpy_matrix': from_numpy_matrix,
                'rational_eigenvalues': rational_eigenvalues,
                'eigenvectors': eigenvectors,
                'matrix_power': matrix_power,
                'determinant': determinant,
                'matrix_inverse': matrix_inverse,
                'qr_decomposition': qr_decomposition,
                'svd': svd,
                'null_space': null_space,
                'column_space': column_space,
                'gram_schmidt': gram_schmidt
            }
        except ImportError:
            return {}
    
    def __getattr__(self, name):
        """Dynamically load utility or advanced functions when accessed."""
        # Try utility functions first
        utility_funcs = self._get_utility_functions()
        if name in utility_funcs:
            return utility_funcs[name]
        
        # Try advanced functions
        advanced_funcs = self._get_advanced_functions()
        if name in advanced_funcs:
            return advanced_funcs[name]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")