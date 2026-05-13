"""
Core linear algebra operations over rational numbers.

Ported from OCaml srk/src/linear.ml and srk/src/ring.ml.

Key conventions (matching OCaml):
  - const_dim = -1  (dimension reserved for the constant 1)
  - Dimensions >= 0 correspond directly to symbol IDs
  - QQVector.pivot does NOT scale (returns (coeff, rest) where add_term(coeff, dim, rest) == original)
  - QQMatrix uses a sparse Dict[int, QQVector] representation internally
  - add_row(i, vec, mat) ADDS vec to row i (not inserts)
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Iterator
from fractions import Fraction
import math

# Type aliases
QQ = Fraction
ZZ = int

# ---------------------------------------------------------------------------
# Affine term conventions  (matching OCaml const_dim = -1)
# ---------------------------------------------------------------------------
const_dim: int = -1


def sym_of_dim(dim: int):
    """Map a dimension to a symbol.  Returns None for const_dim (-1).

    Satisfies: sym_of_dim(dim_of_sym(sym)) == Some(sym)
    Satisfies: sym_of_dim(const_dim) == None
    """
    if dim >= 0:
        from ..core.syntax import symbol_of_int
        return symbol_of_int(dim)
    return None


def dim_of_sym(symbol) -> int:
    """Map a symbol to a dimension (directly uses symbol.id).

    Satisfies: sym_of_dim(dim_of_sym(sym)) == Some(sym)
    """
    if hasattr(symbol, "id"):
        return int(symbol.id)
    if hasattr(symbol, "var_id"):
        return int(symbol.var_id)
    if hasattr(symbol, "name") and symbol.name is not None:
        return hash(symbol.name) % (2**30)
    return 0


def const_linterm(k: QQ) -> QQVector:
    """Representation of a rational number as an affine term."""
    return QQVector.of_term(k, const_dim)


def const_of_linterm(v: QQVector) -> Optional[QQ]:
    """Extract constant from affine term, or None if not constant."""
    coeff, rest = v.pivot(const_dim)
    if rest.is_zero():
        return coeff
    return None


# ---------------------------------------------------------------------------
# QQVector — sparse vector over rationals  (matching OCaml Ring.MakeVector(QQ))
# ---------------------------------------------------------------------------

class QQVector:
    """Sparse vector over QQ with integer dimensions.

    Internal representation: ``entries: Dict[int, QQ]`` mapping dimension to
    coefficient.  Only non-zero entries are stored.

    Matching OCaml semantics:
      - ``pivot(dim)`` returns ``(coeff, rest)`` WITHOUT scaling, such that
        ``add_term(coeff, dim, rest) == original``
      - ``coeff(dim, vec)`` is a static method returning the coefficient
    """

    __slots__ = ("entries",)

    def __init__(self, entries: Optional[Dict[int, QQ]] = None) -> None:
        self.entries: Dict[int, QQ] = entries if entries is not None else {}

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QQVector):
            return False
        return self.entries == other.entries

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.entries.items())))

    # ------------------------------------------------------------------
    # Arithmetic  (all return new QQVector; QQVector is effectively immutable)
    # ------------------------------------------------------------------

    def __add__(self, other: QQVector) -> QQVector:
        result = dict(self.entries)
        for dim, coeff in other.entries.items():
            new_val = result.get(dim, QQ(0)) + coeff
            if new_val == 0:
                result.pop(dim, None)
            else:
                result[dim] = new_val
        return QQVector(result)

    def __sub__(self, other: QQVector) -> QQVector:
        return self + (-other)

    def __neg__(self) -> QQVector:
        return QQVector({d: -c for d, c in self.entries.items()})

    def __mul__(self, scalar: QQ) -> QQVector:
        if scalar == 0:
            return QQVector()
        return QQVector({d: c * scalar for d, c in self.entries.items()})

    def __rmul__(self, scalar: QQ) -> QQVector:
        return self * scalar

    def dot(self, other: QQVector) -> QQ:
        """Inner product: sum_i(self[i] * other[i])."""
        result = QQ(0)
        for dim, coeff in self.entries.items():
            oc = other.entries.get(dim)
            if oc is not None:
                result += coeff * oc
        return result

    def scalar_mul(self, scalar: QQ) -> QQVector:
        """Multiply by scalar (method form for OCaml compatibility)."""
        return self * scalar

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, dim: int, default: QQ = QQ(0)) -> QQ:
        """Get coefficient at dimension *dim*."""
        return self.entries.get(dim, default)

    @staticmethod
    def coeff(dim: int, vec: "QQVector") -> QQ:
        """Get coefficient at *dim* (static, OCaml-compatible)."""
        return vec.entries.get(dim, QQ(0))

    def set(self, dim: int, coeff: QQ) -> QQVector:
        """Return a new vector with *dim* set to *coeff*."""
        new = dict(self.entries)
        if coeff == 0:
            new.pop(dim, None)
        else:
            new[dim] = coeff
        return QQVector(new)

    def add_term(self, coeff: QQ, dim: int) -> QQVector:
        """Return a new vector with *coeff* added to position *dim*."""
        new_val = self.entries.get(dim, QQ(0)) + coeff
        return self.set(dim, new_val)

    def is_zero(self) -> bool:
        return len(self.entries) == 0

    def dimensions(self) -> Set[int]:
        return set(self.entries.keys())

    def dimension(self) -> int:
        """Number of non-zero entries."""
        return len(self.entries)

    # ------------------------------------------------------------------
    # Pivot  (NO scaling — matching OCaml semantics)
    # ------------------------------------------------------------------

    def pivot(self, target_dim: int) -> Tuple[QQ, QQVector]:
        """Extract coefficient at *target_dim* and return ``(coeff, rest)``.

        ``rest`` is ``self`` with the *target_dim* entry removed.
        Invariant: ``add_term(coeff, target_dim, rest) == self``.

        Raises ``KeyError`` if *target_dim* is not present.
        """
        if target_dim not in self.entries:
            raise KeyError(f"Dimension {target_dim} not in QQVector")
        coeff = self.entries[target_dim]
        rest = {d: c for d, c in self.entries.items() if d != target_dim}
        return coeff, QQVector(rest)

    def pop(self) -> Tuple[Tuple[int, QQ], QQVector]:
        """Extract the entry with the smallest dimension and return the rest.

        Returns ``((dim, coeff), rest)``.
        """
        if not self.entries:
            raise ValueError("Cannot pop from zero vector")
        min_dim = min(self.entries.keys())
        coeff = self.entries[min_dim]
        rest = {d: c for d, c in self.entries.items() if d != min_dim}
        return (min_dim, coeff), QQVector(rest)

    # ------------------------------------------------------------------
    # Enumeration / fold / map / merge
    # ------------------------------------------------------------------

    def enum(self) -> List[Tuple[QQ, int]]:
        """Enumerate non-zero entries as ``(coefficient, dimension)`` pairs."""
        return [(c, d) for d, c in self.entries.items()]

    @staticmethod
    def of_enum(pairs) -> QQVector:
        """Build a vector from an iterable of ``(coefficient, dimension)`` pairs."""
        result: Dict[int, QQ] = {}
        for coeff, dim in pairs:
            if coeff != 0:
                new_val = result.get(dim, QQ(0)) + coeff
                if new_val == 0:
                    result.pop(dim, None)
                else:
                    result[dim] = new_val
        return QQVector(result)

    @staticmethod
    def of_list(pairs) -> QQVector:
        return QQVector.of_enum(pairs)

    @staticmethod
    def of_term(coeff: QQ, dim: int) -> QQVector:
        """All-zero vector except *coeff* at *dim*."""
        if coeff == 0:
            return QQVector()
        return QQVector({dim: coeff})

    @staticmethod
    def zero() -> QQVector:
        return QQVector()

    def map(self, f) -> QQVector:
        """Apply ``f(dim, coeff) -> coeff`` to each entry."""
        result: Dict[int, QQ] = {}
        for d, c in self.entries.items():
            new_c = f(d, c)
            if new_c != 0:
                result[d] = new_c
        return QQVector(result)

    def merge(self, f, other: QQVector) -> QQVector:
        """Merge two vectors using ``f(dim, coeff1, coeff2) -> coeff``."""
        all_dims = set(self.entries.keys()) | set(other.entries.keys())
        result: Dict[int, QQ] = {}
        for d in all_dims:
            c1 = self.entries.get(d, QQ(0))
            c2 = other.entries.get(d, QQ(0))
            new_c = f(d, c1, c2)
            if new_c != 0:
                result[d] = new_c
        return QQVector(result)

    def fold(self, f, init):
        """Fold over entries: ``f(dim, coeff, acc) -> acc``."""
        acc = init
        for d, c in self.entries.items():
            acc = f(d, c, acc)
        return acc

    # ------------------------------------------------------------------
    # Interlace / deinterlace  (for pushout)
    # ------------------------------------------------------------------

    @staticmethod
    def interlace(u: QQVector, v: QQVector) -> QQVector:
        """Interlace: u's entries at even positions, v's at odd."""
        result: Dict[int, QQ] = {}
        for coeff, dim in u.enum():
            result[2 * dim] = coeff
        for coeff, dim in v.enum():
            result[2 * dim + 1] = coeff
        return QQVector(result)

    def deinterlace(self) -> Tuple[QQVector, QQVector]:
        """Split into even-indexed and odd-indexed parts."""
        v: Dict[int, QQ] = {}
        w: Dict[int, QQ] = {}
        for coeff, dim in self.enum():
            if dim % 2 == 0:
                v[dim // 2] = coeff
            else:
                w[dim // 2] = coeff
        return QQVector(v), QQVector(w)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(a: QQVector, b: QQVector) -> int:
        """Lexicographic comparison by (dim, coeff)."""
        for d in sorted(set(a.entries.keys()) | set(b.entries.keys())):
            ca = a.entries.get(d, QQ(0))
            cb = b.entries.get(d, QQ(0))
            if ca < cb:
                return -1
            if ca > cb:
                return 1
        return 0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ZZVector — integer-coefficient sparse vector
# ---------------------------------------------------------------------------

class ZZVector:
    """Sparse vector over the integers (matching OCaml ZZVector)."""

    __slots__ = ("entries",)

    def __init__(self, entries: Optional[Dict[int, int]] = None) -> None:
        self.entries: Dict[int, int] = {}
        if entries:
            for dim, coeff in entries.items():
                if coeff != 0:
                    self.entries[dim] = int(coeff)

    def __add__(self, other: ZZVector) -> ZZVector:
        result: Dict[int, int] = dict(self.entries)
        for dim, coeff in other.entries.items():
            new_val = result.get(dim, 0) + coeff
            if new_val == 0:
                result.pop(dim, None)
            else:
                result[dim] = new_val
        return ZZVector(result)

    def __sub__(self, other: ZZVector) -> ZZVector:
        return self + (-other)

    def __neg__(self) -> ZZVector:
        return ZZVector({d: -c for d, c in self.entries.items()})

    def __mul__(self, scalar: int) -> ZZVector:
        if scalar == 0:
            return ZZVector()
        return ZZVector({d: c * scalar for d, c in self.entries.items()})

    def __rmul__(self, scalar: int) -> ZZVector:
        return self * scalar

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZZVector):
            return False
        return self.entries == other.entries

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.entries.items())))

    def get(self, dim: int, default: int = 0) -> int:
        return self.entries.get(dim, default)

    @staticmethod
    def coeff(dim: int, vec: ZZVector) -> int:
        return vec.entries.get(dim, 0)

    def set(self, dim: int, coeff: int) -> ZZVector:
        new = dict(self.entries)
        if coeff == 0:
            new.pop(dim, None)
        else:
            new[dim] = int(coeff)
        return ZZVector(new)

    def add_term(self, coeff: int, dim: int) -> ZZVector:
        return self.set(dim, self.get(dim) + coeff)

    def is_zero(self) -> bool:
        return len(self.entries) == 0

    def dimensions(self) -> Set[int]:
        return set(self.entries.keys())

    def enum(self) -> List[Tuple[int, int]]:
        return [(c, d) for d, c in self.entries.items()]

    def pivot(self, target_dim: int) -> Tuple[int, ZZVector]:
        """Extract coefficient (NO scaling)."""
        if target_dim not in self.entries:
            raise KeyError(f"Dimension {target_dim} not in ZZVector")
        a = self.entries[target_dim]
        rest = {d: c for d, c in self.entries.items() if d != target_dim}
        return a, ZZVector(rest)

    def dot(self, other: ZZVector) -> int:
        result = 0
        for dim, coeff in self.entries.items():
            oc = other.entries.get(dim)
            if oc is not None:
                result += coeff * oc
        return result

    def gcd_normalize(self) -> Tuple[int, ZZVector]:
        if not self.entries:
            return (0, ZZVector())
        g = 0
        for coeff in self.entries.values():
            g = math.gcd(g, abs(coeff))
        if g == 0:
            return (0, ZZVector())
        return (g, ZZVector({d: c // g for d, c in self.entries.items()}))

    def to_qq(self) -> QQVector:
        return QQVector({d: Fraction(c) for d, c in self.entries.items()})

    @staticmethod
    def of_qq(v: QQVector) -> ZZVector:
        result: Dict[int, int] = {}
        for dim, coeff in v.entries.items():
            if coeff.denominator != 1:
                raise ValueError(f"Cannot convert {coeff} to ZZVector")
            if coeff.numerator != 0:
                result[dim] = coeff.numerator
        return ZZVector(result)

    @staticmethod
    def zero() -> ZZVector:
        return ZZVector()

    @staticmethod
    def of_term(coeff: int, dim: int) -> ZZVector:
        return ZZVector({dim: coeff})

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
        return f"ZZVector({dict(self.entries)})"


# ---------------------------------------------------------------------------
# QQMatrix — sparse matrix over rationals  (matching OCaml Ring.MakeMatrix(QQ))
# ---------------------------------------------------------------------------

class QQMatrix:
    """Sparse matrix over QQ.

    Internal representation: ``_rows: Dict[int, QQVector]`` mapping row index
    to row vector (only non-zero rows are stored).

    Matching OCaml semantics:
      - ``add_row(i, vec, mat)`` ADDS *vec* to row *i* (not inserts)
      - ``add_entry(i, j, k, mat)`` ADDS *k* to entry (i, j)
      - All operations are sparse-aware
    """

    __slots__ = ("_rows",)

    def __init__(self, rows=None) -> None:
        if rows is None:
            self._rows: Dict[int, QQVector] = {}
        elif isinstance(rows, dict):
            self._rows = {i: v for i, v in rows.items() if not v.is_zero()}
        elif isinstance(rows, (list, tuple)):
            self._rows = {i: v for i, v in enumerate(rows) if not v.is_zero()}
        else:
            self._rows = {}

    # ------------------------------------------------------------------
    # Backward-compatible ``rows`` property
    # ------------------------------------------------------------------

    @property
    def rows(self) -> Tuple[QQVector, ...]:
        """Padded tuple of rows for backward compatibility."""
        if not self._rows:
            return ()
        max_idx = max(self._rows.keys())
        return tuple(self._rows.get(i, QQVector()) for i in range(max_idx + 1))

    # ------------------------------------------------------------------
    # Equality / hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QQMatrix):
            return False
        return self._rows == other._rows

    @staticmethod
    def equal(m1: QQMatrix, m2: QQMatrix) -> bool:
        return m1._rows == m2._rows

    def __hash__(self) -> int:
        items = tuple(sorted((i, hash(v)) for i, v in self._rows.items()))
        return hash(items)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: QQMatrix) -> QQMatrix:
        new = dict(self._rows)
        for i, row in other._rows.items():
            if i in new:
                s = new[i] + row
                if s.is_zero():
                    del new[i]
                else:
                    new[i] = s
            else:
                new[i] = row
        return QQMatrix(new)

    @staticmethod
    def add(m1: QQMatrix, m2: QQMatrix) -> QQMatrix:
        return m1 + m2

    def __neg__(self) -> QQMatrix:
        return QQMatrix({i: -v for i, v in self._rows.items()})

    def __sub__(self, other: QQMatrix) -> QQMatrix:
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, QQMatrix):
            return self._mat_mul(other)
        elif isinstance(other, QQVector):
            return self._mat_vec_mul(other)
        elif isinstance(other, (int, Fraction)):
            if other == 0:
                return QQMatrix()
            if other == 1:
                return self
            return QQMatrix({i: v * other for i, v in self._rows.items()})
        return NotImplemented

    @staticmethod
    def scalar_mul(scalar: QQ, mat: QQMatrix) -> QQMatrix:
        if scalar == 0:
            return QQMatrix()
        if scalar == 1:
            return mat
        return QQMatrix({i: v * scalar for i, v in mat._rows.items()})

    # ------------------------------------------------------------------
    # Matrix multiplication  (matching OCaml Ring.MakeMatrix(QQ).mul)
    # ------------------------------------------------------------------

    def _mat_mul(self, other: QQMatrix) -> QQMatrix:
        """self * other  (row-col via transpose trick)."""
        other_T = other.transpose()
        result: Dict[int, QQVector] = {}
        for i, row_i in self._rows.items():
            entries: Dict[int, QQ] = {}
            for j, col_j in other_T._rows.items():
                d = row_i.dot(col_j)
                if d != 0:
                    entries[j] = d
            if entries:
                result[i] = QQVector(entries)
        return QQMatrix(result)

    def _mat_vec_mul(self, vector: QQVector) -> QQVector:
        """self * vector.  result[i] = dot(row_i, vector)."""
        entries: Dict[int, QQ] = {}
        for i, row_i in self._rows.items():
            d = row_i.dot(vector)
            if d != 0:
                entries[i] = d
        return QQVector(entries)

    def vector_right_mul(self, vector: QQVector) -> QQVector:
        """m * v  (OCaml-compatible name)."""
        return self._mat_vec_mul(vector)

    def vector_left_mul(self, vector: QQVector) -> QQVector:
        """v^T * m.  result[j] = sum_i(vector[i] * m[i][j])."""
        result = QQVector()
        for i, scalar in vector.entries.items():
            row_i = self._rows.get(i)
            if row_i is not None:
                result = result + row_i.scalar_mul(scalar)
        return result

    # ------------------------------------------------------------------
    # Transpose  (sparse — matching OCaml)
    # ------------------------------------------------------------------

    def transpose(self) -> QQMatrix:
        result: Dict[int, QQVector] = {}
        for i, row_i in self._rows.items():
            for j, coeff in row_i.entries.items():
                if j in result:
                    result[j] = result[j].add_term(coeff, i)
                else:
                    result[j] = QQVector({i: coeff})
        return QQMatrix(result)

    # ------------------------------------------------------------------
    # Matrix exponentiation  (matching OCaml QQMatrix.exp)
    # ------------------------------------------------------------------

    def exp(self, p: int) -> QQMatrix:
        """M^p for p >= 0."""
        dims = sorted(set(self.row_set()) | self.column_set())
        one = QQMatrix.identity(dims)
        result = one
        base = self
        while p > 0:
            if p % 2 == 1:
                result = result * base
            base = base * base
            p //= 2
        return result

    @staticmethod
    def _exp(mat: QQMatrix, p: int) -> QQMatrix:
        return mat.exp(p)

    # ------------------------------------------------------------------
    # Row / column access
    # ------------------------------------------------------------------

    def row(self, dim_or_index, _matrix=None) -> QQVector:
        """Get row at index.  Supports both ``mat.row(i)`` and ``QQMatrix.row(i, mat)``."""
        if _matrix is not None:
            return _matrix._rows.get(dim_or_index, QQVector())
        return self._rows.get(dim_or_index, QQVector())

    def column(self, j: int) -> QQVector:
        """Column *j* as a vector."""
        entries: Dict[int, QQ] = {}
        for i, row_i in self._rows.items():
            c = row_i.entries.get(j)
            if c is not None:
                entries[i] = c
        return QQVector(entries)

    def entry(self, i: int, j: int) -> QQ:
        """Entry at (i, j)."""
        r = self._rows.get(i)
        if r is None:
            return QQ(0)
        return r.entries.get(j, QQ(0))

    def rowsi(self, _matrix=None) -> List[Tuple[int, QQVector]]:
        """Non-zero rows as ``(index, vector)`` pairs.  Supports static call."""
        target = _matrix if _matrix is not None else self
        return sorted(target._rows.items())

    def min_row(self) -> Tuple[int, QQVector]:
        """Row with the smallest index."""
        if not self._rows:
            raise ValueError("Empty matrix")
        min_i = min(self._rows.keys())
        return min_i, self._rows[min_i]

    def row_set(self, _matrix=None) -> Set[int]:
        """Set of row indices with non-zero entries.  Supports static call."""
        target = _matrix if _matrix is not None else self
        return set(target._rows.keys())

    def column_set(self) -> Set[int]:
        cols: Set[int] = set()
        for row in self._rows.values():
            cols.update(row.entries.keys())
        return cols

    def nb_rows(self, _matrix=None) -> int:
        """Number of non-zero rows.  Supports static call."""
        target = _matrix if _matrix is not None else self
        return len(target._rows)

    def nb_columns(self) -> int:
        return len(self.column_set())

    def entries(self):
        """All non-zero entries as ``(i, j, coeff)`` triples."""
        for i, row_i in self._rows.items():
            for j, coeff in row_i.entries.items():
                yield (i, j, coeff)

    # ------------------------------------------------------------------
    # Row / column manipulation
    # ------------------------------------------------------------------

    def add_row(self, dim_or_index, vector=None, _matrix=None) -> QQMatrix:
        """ADD *vector* to row (OCaml semantics: add, not insert).

        Supports both calling conventions:
        - ``mat.add_row(dim, vector)``           (instance method)
        - ``QQMatrix.add_row(dim, vector, mat)`` (static-style)
        """
        if vector is not None and _matrix is not None:
            # Static-style: QQMatrix.add_row(dim, vector, mat)
            dim = dim_or_index
            mat = _matrix
            existing = mat._rows.get(dim, QQVector())
            new_row = existing + vector
            new_rows = dict(mat._rows)
            if new_row.is_zero():
                new_rows.pop(dim, None)
            else:
                new_rows[dim] = new_row
            return QQMatrix(new_rows)
        elif vector is not None:
            # Instance-style: mat.add_row(dim, vector)
            dim = dim_or_index
            existing = self._rows.get(dim, QQVector())
            new_row = existing + vector
            new_rows = dict(self._rows)
            if new_row.is_zero():
                new_rows.pop(dim, None)
            else:
                new_rows[dim] = new_row
            return QQMatrix(new_rows)
        else:
            raise TypeError("add_row requires at least 2 arguments")

    def add_column(self, j: int, vec: QQVector) -> QQMatrix:
        """ADD *vec* as column *j*."""
        new_rows = dict(self._rows)
        for i, coeff in vec.entries.items():
            existing = new_rows.get(i, QQVector())
            new_rows[i] = existing.add_term(coeff, j)
        return QQMatrix(new_rows)

    def add_entry(self, i: int, j: int, k: QQ) -> QQMatrix:
        """ADD *k* to entry (i, j)."""
        if k == 0:
            return self
        existing = self._rows.get(i, QQVector())
        return self.add_row(i, QQVector({j: k}))

    def pivot(self, dim: int) -> Tuple[QQVector, QQMatrix]:
        """Extract row *dim* and remove it."""
        row = self._rows.get(dim, QQVector())
        new_rows = {i: v for i, v in self._rows.items() if i != dim}
        return row, QQMatrix(new_rows)

    def pivot_column(self, j: int) -> Tuple[QQVector, QQMatrix]:
        """Extract column *j* and remove it."""
        column_entries: Dict[int, QQ] = {}
        new_rows: Dict[int, QQVector] = {}
        for i, row_i in self._rows.items():
            coeff = row_i.entries.get(j)
            if coeff is not None:
                column_entries[i] = coeff
            rest = {d: c for d, c in row_i.entries.items() if d != j}
            if rest:
                new_rows[i] = QQVector(rest)
        return QQVector(column_entries), QQMatrix(new_rows)

    def map_rows(self, f) -> QQMatrix:
        """Apply *f(row) -> new_row* to each non-zero row."""
        new_rows: Dict[int, QQVector] = {}
        for i, row_i in self._rows.items():
            new_row = f(row_i)
            if not new_row.is_zero():
                new_rows[i] = new_row
        return QQMatrix(new_rows)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def zero() -> QQMatrix:
        return QQMatrix()

    @staticmethod
    def identity(dimensions: List[int]) -> QQMatrix:
        """Identity restricted to *dimensions*."""
        rows: Dict[int, QQVector] = {}
        for d in dimensions:
            rows[d] = QQVector({d: QQ(1)})
        return QQMatrix(rows)

    @staticmethod
    def of_rows(vecs: List[QQVector]) -> QQMatrix:
        """Matrix whose row *i* is *vecs[i]*."""
        rows: Dict[int, QQVector] = {}
        for i, v in enumerate(vecs):
            if not v.is_zero():
                rows[i] = v
        return QQMatrix(rows)

    @staticmethod
    def of_dense(dense) -> QQMatrix:
        """From a 2D list/array."""
        rows: Dict[int, QQVector] = {}
        for i, dense_row in enumerate(dense):
            entries: Dict[int, QQ] = {}
            for j, val in enumerate(dense_row):
                v = QQ(val) if not isinstance(val, QQ) else val
                if v != 0:
                    entries[j] = v
            if entries:
                rows[i] = QQVector(entries)
        return QQMatrix(rows)

    def dense_of(self, num_rows: int, num_cols: int):
        """To a dense 2D list."""
        result = [[QQ(0)] * num_cols for _ in range(num_rows)]
        for i, row_i in self._rows.items():
            if i < num_rows:
                for j, coeff in row_i.entries.items():
                    if j < num_cols:
                        result[i][j] = coeff
        return result

    # ------------------------------------------------------------------
    # Interlace columns  (for pushout)
    # ------------------------------------------------------------------

    @staticmethod
    def interlace_columns(m: QQMatrix, n: QQMatrix) -> QQMatrix:
        """Interlace columns: even cols from *m*, odd cols from *n*."""
        all_rows = set(m._rows.keys()) | set(n._rows.keys())
        result: Dict[int, QQVector] = {}
        for i in all_rows:
            row_m = m._rows.get(i, QQVector())
            row_n = n._rows.get(i, QQVector())
            interlaced = QQVector.interlace(row_m, row_n)
            if not interlaced.is_zero():
                result[i] = interlaced
        return QQMatrix(result)

    # ------------------------------------------------------------------
    # Rank  (Gaussian elimination)
    # ------------------------------------------------------------------

    def rank(self) -> int:
        if not self._rows:
            return 0
        # Work on a mutable copy
        working = {i: dict(v.entries) for i, v in self._rows.items()}
        all_cols = self.column_set()
        if not all_cols:
            return 0
        rank = 0
        for col in sorted(all_cols):
            # Find pivot
            pivot_row = None
            for r in sorted(working.keys()):
                if working[r].get(col, QQ(0)) != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            # Eliminate
            pivot_coeff = working[pivot_row][col]
            for r in sorted(working.keys()):
                if r == pivot_row:
                    continue
                factor = working[r].get(col, QQ(0))
                if factor != 0:
                    scale = factor / pivot_coeff
                    for d, c in working[pivot_row].items():
                        working[r][d] = working[r].get(d, QQ(0)) - c * scale
                    # Clean zeros
                    working[r] = {d: c for d, c in working[r].items() if c != 0}
            rank += 1
        return rank

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> QQMatrix:
        return QQMatrix({i: QQVector(dict(v.entries)) for i, v in self._rows.items()})

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if not self._rows:
            return "[]"
        return "\n".join(str(row) for _, row in self.rowsi())

    def __repr__(self) -> str:
        return f"QQMatrix({dict(self._rows)})"

    # ------------------------------------------------------------------
    # Rational eigenvalues  (porting OCaml QQMatrix.rational_eigenvalues)
    # ------------------------------------------------------------------

    def rational_eigenvalues(self, dims: List[int]) -> List[Tuple[QQ, int]]:
        """Compute rational eigenvalues with algebraic multiplicities."""
        if not dims:
            return []
        # Scale to integer matrix
        denom = ZZ(1)
        for i in dims:
            for j in dims:
                e = self.entry(i, j)
                denom = math.lcm(denom, e.denominator)
        n = len(dims)
        int_mat = [[0] * n for _ in range(n)]
        for ii, di in enumerate(dims):
            for jj, dj in enumerate(dims):
                e = self.entry(di, dj)
                int_mat[ii][jj] = int(e * denom)
        # Use sympy for characteristic polynomial factoring
        try:
            import sympy
            x = sympy.Symbol("x")
            M = sympy.Matrix(int_mat)
            cp = M.charpoly(x)
            factors = sympy.factor_list(cp.as_expr(), x)
            result: List[Tuple[QQ, int]] = []
            for factor, mult in factors[1]:
                if sympy.degree(factor, x) == 1:
                    coeffs = sympy.Poly(factor, x).all_coeffs()
                    a_val = int(coeffs[0])
                    b_val = int(coeffs[1])
                    eigenvalue = QQ(-b_val, a_val * denom)
                    result.append((eigenvalue, mult))
            return result
        except (ImportError, Exception):
            return []


# ---------------------------------------------------------------------------
# Row echelon form / nullspace / solve  (porting OCaml)
# ---------------------------------------------------------------------------

class NoSolution(Exception):
    pass


def _row_echelon_form(mat: QQMatrix, b_column: int) -> List[Tuple[int, QQVector]]:
    """Gaussian elimination returning ``(pivot_col, scaled_row)`` pairs.

    Each pair represents the equation ``pivot_col = dot(soln, scaled_row)``.
    """
    working = {i: QQVector(dict(v.entries)) for i, v in mat._rows.items()}
    finished: List[Tuple[int, QQVector]] = []

    while working:
        # Find min row
        row_num = min(working.keys())
        next_row = working.pop(row_num)
        # Find first column != b_column
        column = None
        for _, dim in next_row.enum():
            if dim != b_column:
                column = dim
                break
        if column is None:
            raise NoSolution("Singular system")
        # Pivot: extract coefficient and rest
        cell = next_row.entries.get(column, QQ(0))
        rest = {d: c for d, c in next_row.entries.items() if d != column}
        rest_vec = QQVector(rest)
        # Scale: next_row' = -(1/cell) * rest  (so that column = dot(soln, next_row'))
        neg_inv = -QQ(1) / cell
        scaled = rest_vec * neg_inv
        # Eliminate column from remaining rows
        new_working: Dict[int, QQVector] = {}
        for r, row_r in working.items():
            coeff_r = row_r.entries.get(column, QQ(0))
            if coeff_r != 0:
                # row_r' = row_r_rest + coeff_r * scaled
                rest_r = {d: c for d, c in row_r.entries.items() if d != column}
                new_row = QQVector(rest_r) + scaled * coeff_r
                if not new_row.is_zero():
                    new_working[r] = new_row
            else:
                new_working[r] = row_r
        working = new_working
        finished.append((column, scaled))

    # Reverse to match OCaml order (last-processed row first) for backpropagation
    finished.reverse()
    return finished


def nullspace(mat: QQMatrix, dimensions: List[int]) -> List[QQVector]:
    """Nullspace of *mat* projected onto *dimensions*."""
    columns = mat.column_set()
    b_column = 1 + (max(columns) if columns else 0)
    b_column = max(b_column, max(dimensions) + 1 if dimensions else 0)
    rr = _row_echelon_form(mat, b_column)

    def backprop(soln: QQVector, pairs: List[Tuple[int, QQVector]]) -> QQVector:
        for lhs, rhs in pairs:
            soln = soln.add_term(soln.dot(rhs), lhs)
        return soln

    pivot_cols = {col for col, _ in rr}
    free_dims = [d for d in dimensions if d not in pivot_cols]
    return [backprop(QQVector.of_term(QQ(1), d), rr) for d in free_dims]


def solve_exn(mat: QQMatrix, b: QQVector) -> QQVector:
    """Solve mat * x = b.  Raises NoSolution if unsolvable."""
    columns = mat.column_set()
    b_column = 1 + (max(columns) if columns else 0)
    augmented = mat.add_column(b_column, b)
    rr = _row_echelon_form(augmented, b_column)

    def backprop(soln: QQVector, pairs: List[Tuple[int, QQVector]]) -> QQVector:
        for lhs, rhs in pairs:
            soln = soln.add_term(soln.dot(rhs), lhs)
        return soln

    soln = backprop(QQVector.of_term(QQ(1), b_column), rr)
    # Extract solution: pivot out b_column, negate
    b_coeff = soln.entries.get(b_column, QQ(0))
    rest = {d: c for d, c in soln.entries.items() if d != b_column}
    return -QQVector(rest)


def solve(mat: QQMatrix, b: QQVector) -> Optional[QQVector]:
    try:
        return solve_exn(mat, b)
    except NoSolution:
        return None


# ---------------------------------------------------------------------------
# Vector space operations  (porting OCaml QQVectorSpace)
# ---------------------------------------------------------------------------

def _mem_vector_space(basis: List[QQVector], v: QQVector) -> bool:
    """Check if *v* is in the span of *basis*."""
    if not basis:
        return v.is_zero()
    mA = QQMatrix.of_rows(basis)
    return solve(mA.transpose(), v) is not None


class QQVectorSpace:
    """Vector space represented by a list of basis vectors.

    Matching OCaml ``QQVectorSpace`` module.
    """

    def __init__(self, basis: List[QQVector]):
        self.basis = basis

    @property
    def _t(self) -> List[QQVector]:
        return self.basis

    @staticmethod
    def empty() -> QQVectorSpace:
        return QQVectorSpace([])

    def is_empty(self) -> bool:
        return len(self.basis) == 0

    def mem(self, v: QQVector) -> bool:
        return _mem_vector_space(self.basis, v)

    def subspace(self, other: QQVectorSpace) -> bool:
        return all(_mem_vector_space(other.basis, v) for v in self.basis)

    def equal(self, other: QQVectorSpace) -> bool:
        return self.subspace(other) and other.subspace(self)

    def matrix_of(self) -> QQMatrix:
        return QQMatrix.of_rows(self.basis)

    @staticmethod
    def of_matrix(m: QQMatrix) -> QQVectorSpace:
        return QQVectorSpace([row for _, row in m.rowsi()])

    def intersect(self, other: QQVectorSpace) -> QQVectorSpace:
        mU = self.matrix_of()
        mV = other.matrix_of()
        mC, _ = intersect_rowspace(mU, mV)
        return QQVectorSpace.of_matrix(mC * mU)

    def sum(self, other: QQVectorSpace) -> QQVectorSpace:
        result = list(self.basis)
        for v in other.basis:
            if not _mem_vector_space(result, v):
                result.append(v)
        return QQVectorSpace(result)

    def diff(self, other: QQVectorSpace) -> QQVectorSpace:
        result: List[QQVector] = []
        combined = result + other.basis
        for v in self.basis:
            if not _mem_vector_space(combined, v):
                result.append(v)
                combined = result + other.basis
        return QQVectorSpace(result)

    @staticmethod
    def standard_basis(dim: int) -> QQVectorSpace:
        return QQVectorSpace([QQVector.of_term(QQ(1), d) for d in range(dim)])

    @staticmethod
    def basis_from_vectors(vecs: List[QQVector]) -> QQVectorSpace:
        result: List[QQVector] = []
        for v in vecs:
            if not _mem_vector_space(result, v):
                result.append(v)
        return QQVectorSpace(result)

    def simplify(self) -> QQVectorSpace:
        """Gauss-Jordan simplification."""
        basis = list(self.basis)
        result: List[QQVector] = []
        remaining = list(basis)
        while remaining:
            y = remaining.pop(0)
            if y.is_zero():
                continue
            # Find leading dimension
            min_dim = min(y.entries.keys())
            coeff = y.entries[min_dim]
            # Normalize so leading coefficient is 1
            inv = QQ(1) / coeff
            y_norm = y * inv
            # Eliminate leading dimension from all other vectors
            def reduce_vec(x: QQVector) -> QQVector:
                c = x.entries.get(min_dim, QQ(0))
                if c == 0:
                    return x
                return x + y_norm * (-c)

            result = [reduce_vec(x) for x in result]
            remaining = [reduce_vec(x) for x in remaining]
            result.append(y_norm)
        return QQVectorSpace(result)

    def scale_integer(self) -> QQVectorSpace:
        scaled = []
        for vec in self.basis:
            lcm_denom = 1
            for coeff in vec.entries.values():
                lcm_denom = math.lcm(lcm_denom, coeff.denominator)
            scaled.append(vec * QQ(lcm_denom))
        return QQVectorSpace(scaled)

    def dimension(self) -> int:
        return len(self.basis)

    def contains(self, vector: QQVector) -> bool:
        return self.mem(vector)

    @staticmethod
    def equal_static(s1: QQVectorSpace, s2: QQVectorSpace) -> bool:
        return s1.equal(s2)

    def __str__(self) -> str:
        return f"VectorSpace(dimension={self.dimension()})"


# ---------------------------------------------------------------------------
# intersect_rowspace  (porting OCaml)
# ---------------------------------------------------------------------------

def intersect_rowspace(a: QQMatrix, b: QQMatrix) -> Tuple[QQMatrix, QQMatrix]:
    """Compute C, D such that C*A = D*B is a basis for rowspace(A) ∩ rowspace(B)."""
    # Build lambda_1*A - lambda_2*B = 0 system
    # lambda_1 at even columns, lambda_2 at odd columns
    mat_a = QQMatrix()
    for i, j, k in a.entries():
        mat_a = mat_a.add_entry(j, 2 * i, k)

    mat = mat_a
    for i, j, k in b.entries():
        mat = mat.add_entry(j, 2 * i + 1, -k)

    c = QQMatrix()
    d = QQMatrix()
    c_rows = 0
    d_rows = 0
    mat_rows = max(r for r, _ in mat.rowsi()) + 1 if mat._rows else 0

    # For each column in the row space, try to find a vector in the intersection
    # with 1 in that column's entry
    for col, _ in mat.rowsi():
        # Add constraint: row col of mat_a must match
        mat_prime = mat.add_row(mat_rows, mat_a.row(col))
        target = QQVector.of_term(QQ(1), mat_rows)
        solution = solve(mat_prime, target)
        if solution is not None:
            c_row = QQVector()
            d_row = QQVector()
            for entry, i in solution.enum():
                if i % 2 == 0:
                    c_row = c_row.add_term(entry, i // 2)
                else:
                    d_row = d_row.add_term(entry, i // 2)
            c = c.add_row(c_rows, c_row)
            d = d.add_row(d_rows, d_row)
            c_rows += 1
            d_rows += 1
            mat_rows += 1
            mat = mat_prime

    return c, d


# ---------------------------------------------------------------------------
# pushout  (porting OCaml)
# ---------------------------------------------------------------------------

def pushout(mA: QQMatrix, mB: QQMatrix) -> Tuple[QQMatrix, QQMatrix]:
    """Pushout in the category of rational vector spaces.

    Returns (C, D) such that C*A = D*B, and for any E, F with E*A = F*B,
    there exists unique U with U*C*A = U*D*B = E*A = F*B.
    """
    mABt = QQMatrix.interlace_columns(mA.transpose(), QQMatrix.scalar_mul(QQ(-1), mB))
    pairs = nullspace(mABt, sorted(mABt.column_set()))
    mC = QQMatrix()
    mD = QQMatrix()
    for i, soln in enumerate(pairs):
        c, d = soln.deinterlace()
        mC = mC.add_row(i, c)
        mD = mD.add_row(i, d)
    return mC, mD


# ---------------------------------------------------------------------------
# divide_right / divide_left  (porting OCaml)
# ---------------------------------------------------------------------------

def divide_right(a: QQMatrix, b: QQMatrix) -> Optional[QQMatrix]:
    """Find C such that C*B = A (i.e., rowspace(B) ⊆ rowspace(A))."""
    try:
        b_tr = b.transpose()
        div = QQMatrix()
        for i, row in a.rowsi():
            sol = solve_exn(b_tr, row)
            div = div.add_row(i, sol)
        return div
    except NoSolution:
        return None


def divide_left(a: QQMatrix, b: QQMatrix) -> Optional[QQMatrix]:
    """Find C such that B*C = A."""
    result = divide_right(a.transpose(), b.transpose())
    return result.transpose() if result is not None else None


# ---------------------------------------------------------------------------
# Spectral decomposition  (porting OCaml)
# ---------------------------------------------------------------------------

def rational_spectral_decomposition(mA: QQMatrix, dims: List[int]) -> List[Tuple[QQ, QQVector]]:
    """(eigenvalue, generalized eigenvector) pairs."""
    mAt = mA.transpose()
    identity = QQMatrix.identity(dims)
    rsd: List[Tuple[QQ, QQVector]] = []
    for lam, mult in mA.rational_eigenvalues(dims):
        mE = (mAt + QQMatrix.scalar_mul(-lam, identity)).exp(mult)
        for v in nullspace(mE, dims):
            rsd.append((lam, v))
    return rsd


def periodic_rational_spectral_decomposition(
    mA: QQMatrix, dims: List[int]
) -> List[Tuple[int, QQ, QQVector]]:
    """(period, eigenvalue, generalized eigenvector) triples."""
    nb_dims = len(dims)
    max_pow = nb_dims * nb_dims * nb_dims
    prsd: List[Tuple[int, QQ, QQVector]] = []
    mA_pow = mA
    for i in range(1, max_pow + 1):
        if len(prsd) == nb_dims:
            break
        existing_space = [v for _, _, v in prsd]
        for lam, v in rational_spectral_decomposition(mA_pow, dims):
            if not _mem_vector_space(existing_space, v):
                prsd.append((i, lam, v))
                existing_space.append(v)
        mA_pow = mA * mA_pow
    return prsd


def jordan_chain(mA: QQMatrix, lam: QQ, v: QQVector) -> List[QQVector]:
    """Compute left Jordan chain for eigenvalue *lam* starting from *v*."""
    residual = mA.vector_left_mul(v) - v * lam
    if residual.is_zero():
        return [v]
    return [v] + jordan_chain(mA, lam, residual)


# ---------------------------------------------------------------------------
# Affine term functions  (porting OCaml)
# ---------------------------------------------------------------------------

exception_Nonlinear = ValueError("Nonlinear term encountered in linterm_of")


def linterm_of(*args) -> QQVector:
    """Extract a linear term (QQVector) from a syntax expression.

    Supports both ``linterm_of(expr)`` and ``linterm_of(srk, expr)``.
    """
    if len(args) == 1:
        expr = args[0]
    elif len(args) == 2:
        _, expr = args
    else:
        raise TypeError(f"linterm_of expects (expr) or (srk, expr), got {len(args)} args")

    from ..core.syntax import Var, Const, Add, Mul

    def _const_as_fraction(c: Const) -> Optional[QQ]:
        name = getattr(c.symbol, "name", None)
        if not name:
            return None
        try:
            return QQ(name)
        except Exception:
            pass
        if name.startswith("real_"):
            try:
                return QQ(str(float(name[5:])))
            except Exception:
                return None
        return None

    def _scale(v: QQVector, k: QQ) -> QQVector:
        if k == 0 or not v.entries:
            return QQVector()
        return QQVector({d: c * k for d, c in v.entries.items() if c * k != 0})

    def _add(vs: list) -> QQVector:
        out: Dict[int, QQ] = {}
        for v in vs:
            for d, c in v.entries.items():
                out[d] = out.get(d, QQ(0)) + c
                if out[d] == 0:
                    del out[d]
        return QQVector(out)

    def rec(e) -> QQVector:
        if isinstance(e, Var):
            return QQVector.of_term(QQ(1), dim_of_sym(e))
        if isinstance(e, Const):
            k = _const_as_fraction(e)
            if k is not None:
                return QQVector.of_term(k, const_dim) if k != 0 else QQVector()
            return QQVector.of_term(QQ(1), dim_of_sym(e.symbol))
        if isinstance(e, Add):
            return _add([rec(a) for a in e.args])
        if isinstance(e, Mul):
            scalar = QQ(1)
            non_scalar = []
            for a in e.args:
                if isinstance(a, Const):
                    k = _const_as_fraction(a)
                    if k is not None:
                        scalar *= k
                        continue
                non_scalar.append(a)
            if not non_scalar:
                return QQVector.of_term(scalar, const_dim) if scalar != 0 else QQVector()
            if len(non_scalar) == 1:
                return _scale(rec(non_scalar[0]), scalar)
            raise ValueError("Nonlinear term encountered in linterm_of")
        raise ValueError(f"Unsupported term in linterm_of: {type(e)}")

    return rec(expr)


def of_linterm(srk, linterm: QQVector):
    """Convert a QQVector to a syntax arithmetic term."""
    from ..core.syntax import mk_real, mk_const, mk_mul, mk_add, symbol_of_int

    parts = []
    for coeff, dim in linterm.enum():
        sym = sym_of_dim(dim)
        if sym is not None:
            if QQ(coeff) == QQ(1):
                parts.append(mk_const(srk, sym))
            else:
                parts.append(mk_mul(srk, [mk_real(srk, coeff), mk_const(srk, sym)]))
        else:
            parts.append(mk_real(srk, coeff))
    if not parts:
        return mk_real(srk, QQ(0))
    return mk_add(srk, parts)


def evaluate_linterm(interp, term: QQVector) -> QQ:
    """Evaluate affine term using symbol interpretation."""
    result = QQ(0)
    for coeff, dim in term.enum():
        sym = sym_of_dim(dim)
        if sym is not None:
            result += interp(sym) * coeff
        else:
            result += coeff
    return result


def evaluate_affine(m, term: QQVector) -> QQ:
    """Evaluate affine term using dimension-based interpretation."""
    result = QQ(0)
    for coeff, dim in term.enum():
        if dim == const_dim:
            result += coeff
        else:
            result += m(dim) * coeff
    return result


def linterm_size(linterm: QQVector) -> int:
    """Number of non-zero dimensions."""
    return len(linterm.entries)


def term_of_vec(srk, term_of_dim, vec: QQVector):
    """Create a term by interpreting each dimension with *term_of_dim*."""
    from ..core.syntax import mk_real, mk_mul, mk_add

    parts = []
    for coeff, dim in vec.enum():
        parts.append(mk_mul(srk, [mk_real(srk, coeff), term_of_dim(dim)]))
    if not parts:
        return mk_real(srk, QQ(0))
    return mk_add(srk, parts)


def pp_linterm(srk, formatter, linterm: QQVector):
    """Pretty-print an affine term (stub)."""
    pass


# ---------------------------------------------------------------------------
# Utility function wrappers  (backward compatibility)
# ---------------------------------------------------------------------------

def zero_vector(dimensions: int) -> QQVector:
    return QQVector()


def unit_vector(dim: int, size: int) -> QQVector:
    return QQVector.of_term(QQ(1), dim)


def identity_matrix(size: int) -> QQMatrix:
    return QQMatrix.identity(list(range(size)))


def vector_from_list(values) -> QQVector:
    entries = {i: QQ(v) for i, v in enumerate(values) if QQ(v) != 0}
    return QQVector(entries)


def matrix_from_lists(rows) -> QQMatrix:
    return QQMatrix.of_rows([vector_from_list(row) for row in rows])


def mk_vector(values) -> QQVector:
    return vector_from_list(values)


def mk_matrix(rows) -> QQMatrix:
    return matrix_from_lists(rows)


def solve_linear_system(matrix: QQMatrix, vector: QQVector) -> Optional[QQVector]:
    return solve(matrix, vector)


# ---------------------------------------------------------------------------
# Advanced functions  (lazy imports from linear_advanced)
# ---------------------------------------------------------------------------

def to_numpy_matrix(matrix):
    from .linear_advanced import to_numpy_matrix as _f
    return _f(matrix)


def from_numpy_matrix(arr):
    from .linear_advanced import from_numpy_matrix as _f
    return _f(arr)


def rational_eigenvalues_fn(matrix):
    from .linear_advanced import rational_eigenvalues as _f
    return _f(matrix)


def eigenvectors(matrix):
    from .linear_advanced import eigenvectors as _f
    return _f(matrix)


def matrix_power(matrix, n):
    from .linear_advanced import matrix_power as _f
    return _f(matrix, n)


def determinant(matrix):
    from .linear_advanced import determinant as _f
    return _f(matrix)


def matrix_inverse(matrix):
    from .linear_advanced import matrix_inverse as _f
    return _f(matrix)


def qr_decomposition(matrix):
    from .linear_advanced import qr_decomposition as _f
    return _f(matrix)


def svd(matrix):
    from .linear_advanced import svd as _f
    return _f(matrix)


def null_space(matrix):
    from .linear_advanced import null_space as _f
    return _f(matrix)


def column_space(matrix):
    from .linear_advanced import column_space as _f
    return _f(matrix)


def gram_schmidt(vectors):
    from .linear_advanced import gram_schmidt as _f
    return _f(vectors)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "QQVector",
    "QQMatrix",
    "QQVectorSpace",
    "ZZVector",
    # Conventions
    "const_dim",
    "dim_of_sym",
    "sym_of_dim",
    "const_linterm",
    "const_of_linterm",
    # Solvers
    "solve",
    "solve_exn",
    "nullspace",
    "intersect_rowspace",
    "pushout",
    "divide_right",
    "divide_left",
    # Spectral
    "rational_spectral_decomposition",
    "periodic_rational_spectral_decomposition",
    "jordan_chain",
    # Affine terms
    "linterm_of",
    "of_linterm",
    "evaluate_linterm",
    "evaluate_affine",
    "linterm_size",
    "term_of_vec",
    # Utilities
    "zero_vector",
    "unit_vector",
    "identity_matrix",
    "vector_from_list",
    "matrix_from_lists",
    "mk_vector",
    "mk_matrix",
    "solve_linear_system",
    # Advanced
    "to_numpy_matrix",
    "from_numpy_matrix",
    "rational_eigenvalues_fn",
    "eigenvectors",
    "matrix_power",
    "determinant",
    "matrix_inverse",
    "qr_decomposition",
    "svd",
    "null_space",
    "column_space",
    "gram_schmidt",
]


# ---------------------------------------------------------------------------
# Linear namespace  (backward compatibility)
# ---------------------------------------------------------------------------

class Linear:
    """Namespace for linear algebra functions."""

    QQVector = QQVector
    QQMatrix = QQMatrix
    QQVectorSpace = QQVectorSpace
    ZZVector = ZZVector
    const_dim = const_dim

    @staticmethod
    def dim_of_sym(symbol) -> int:
        return dim_of_sym(symbol)

    @staticmethod
    def _get_utility_functions():
        return {
            "zero_vector": zero_vector,
            "unit_vector": unit_vector,
            "identity_matrix": identity_matrix,
            "vector_from_list": vector_from_list,
            "matrix_from_lists": matrix_from_lists,
            "mk_vector": mk_vector,
            "mk_matrix": mk_matrix,
            "solve_linear_system": solve_linear_system,
            "linterm_of": linterm_of,
        }

    @staticmethod
    def _get_advanced_functions():
        try:
            return {
                "to_numpy_matrix": to_numpy_matrix,
                "from_numpy_matrix": from_numpy_matrix,
                "rational_eigenvalues": rational_eigenvalues_fn,
                "eigenvectors": eigenvectors,
                "matrix_power": matrix_power,
                "determinant": determinant,
                "matrix_inverse": matrix_inverse,
                "qr_decomposition": qr_decomposition,
                "svd": svd,
                "null_space": null_space,
                "column_space": column_space,
                "gram_schmidt": gram_schmidt,
            }
        except ImportError:
            return {}

    def __getattr__(self, name):
        utility = self._get_utility_functions()
        if name in utility:
            return utility[name]
        advanced = self._get_advanced_functions()
        if name in advanced:
            return advanced[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
