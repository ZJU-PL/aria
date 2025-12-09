"""
Advanced linear algebra operations using numpy.

This module provides additional linear algebra functionality that requires
numpy/scipy for numerical computations, including eigenvalues, matrix
decompositions, and other advanced operations.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from fractions import Fraction

# Optional numpy import for advanced operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from aria.srk.linear import QQMatrix, QQVector, QQ


def to_numpy_matrix(matrix: QQMatrix) -> 'np.ndarray':
    """Convert QQMatrix to numpy array."""
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return np.array([[]])

    # Get dimensions
    all_cols = set()
    for row in matrix.rows:
        all_cols.update(row.dimensions())

    if not all_cols:
        return np.zeros((len(matrix.rows), 0))

    max_col = max(all_cols) + 1

    # Build numpy array
    arr = np.zeros((len(matrix.rows), max_col))
    for i, row in enumerate(matrix.rows):
        for j in range(max_col):
            coeff = row.get(j, QQ(0))
            arr[i, j] = float(coeff)

    return arr


def from_numpy_matrix(arr: 'np.ndarray') -> QQMatrix:
    """Convert numpy array to QQMatrix."""
    rows = []
    for i in range(arr.shape[0]):
        entries = {}
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if abs(val) > 1e-10:  # Threshold for zero
                entries[j] = Fraction(val).limit_denominator(10000)
        rows.append(QQVector(entries))

    return QQMatrix(rows)


def rational_eigenvalues(matrix: QQMatrix) -> List[Fraction]:
    """
    Compute rational eigenvalues of a matrix.

    This function computes eigenvalues and returns those that are
    (approximately) rational numbers.

    Args:
        matrix: The matrix to compute eigenvalues for

    Returns:
        List of rational eigenvalues
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return []

    # Convert to numpy
    arr = to_numpy_matrix(matrix)

    # Check if square
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square to compute eigenvalues")

    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(arr)

    # Filter to (approximately) real eigenvalues and convert to rational
    rational_eigs = []
    for eig in eigenvals:
        if abs(eig.imag) < 1e-10:  # Essentially real
            real_val = eig.real
            # Convert to fraction with limited denominator
            frac = Fraction(real_val).limit_denominator(10000)
            rational_eigs.append(frac)

    return rational_eigs


def eigenvectors(matrix: QQMatrix) -> List[Tuple[Fraction, QQVector]]:
    """
    Compute eigenvectors and corresponding eigenvalues.

    Returns a list of (eigenvalue, eigenvector) pairs.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return []

    arr = to_numpy_matrix(matrix)

    # Check if square
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square to compute eigenvectors")

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(arr)

    result = []
    for i in range(len(eigenvals)):
        eig = eigenvals[i]
        vec = eigenvecs[:, i]

        # Only include if eigenvalue is approximately real
        if abs(eig.imag) < 1e-10:
            # Convert eigenvalue to fraction
            frac_eig = Fraction(eig.real).limit_denominator(10000)

            # Convert eigenvector to QQVector
            vec_entries = {}
            for j in range(len(vec)):
                if abs(vec[j].real) > 1e-10:
                    vec_entries[j] = Fraction(vec[j].real).limit_denominator(10000)

            result.append((frac_eig, QQVector(vec_entries)))

    return result


def matrix_power(matrix: QQMatrix, n: int) -> QQMatrix:
    """
    Compute matrix raised to power n.

    Args:
        matrix: The matrix to raise to a power
        n: The exponent (must be non-negative)

    Returns:
        matrix^n
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if n < 0:
        raise ValueError("Negative powers not supported")

    if n == 0:
        # Return identity matrix
        size = len(matrix.rows)
        rows = []
        for i in range(size):
            entries = {i: QQ(1)}
            rows.append(QQVector(entries))
        return QQMatrix(rows)

    if n == 1:
        return matrix

    # Use numpy for efficient computation
    arr = to_numpy_matrix(matrix)

    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square for matrix power")

    result_arr = np.linalg.matrix_power(arr, n)

    return from_numpy_matrix(result_arr)


def determinant(matrix: QQMatrix) -> Fraction:
    """
    Compute the determinant of a matrix.

    Args:
        matrix: The matrix to compute determinant for

    Returns:
        The determinant as a rational number
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return QQ(1)

    arr = to_numpy_matrix(matrix)

    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square to compute determinant")

    det = np.linalg.det(arr)

    return Fraction(det).limit_denominator(10000)


def matrix_inverse(matrix: QQMatrix) -> Optional[QQMatrix]:
    """
    Compute the inverse of a matrix if it exists.

    Args:
        matrix: The matrix to invert

    Returns:
        The inverse matrix, or None if the matrix is singular
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return None

    arr = to_numpy_matrix(matrix)

    if arr.shape[0] != arr.shape[1]:
        return None  # Not square

    try:
        inv_arr = np.linalg.inv(arr)
        return from_numpy_matrix(inv_arr)
    except np.linalg.LinAlgError:
        return None  # Singular matrix


def qr_decomposition(matrix: QQMatrix) -> Tuple[QQMatrix, QQMatrix]:
    """
    Compute QR decomposition of a matrix.

    Returns (Q, R) where matrix = Q * R, Q is orthogonal, and R is upper triangular.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return QQMatrix([]), QQMatrix([])

    arr = to_numpy_matrix(matrix)

    Q, R = np.linalg.qr(arr)

    return from_numpy_matrix(Q), from_numpy_matrix(R)


def svd(matrix: QQMatrix) -> Tuple[QQMatrix, List[Fraction], QQMatrix]:
    """
    Compute Singular Value Decomposition.

    Returns (U, singular_values, Vt) where matrix = U * diag(singular_values) * Vt
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return QQMatrix([]), [], QQMatrix([])

    arr = to_numpy_matrix(matrix)

    U, s, Vt = np.linalg.svd(arr, full_matrices=False)

    # Convert singular values to fractions
    singular_values = [Fraction(val).limit_denominator(10000) for val in s]

    return from_numpy_matrix(U), singular_values, from_numpy_matrix(Vt)


def null_space(matrix: QQMatrix) -> List[QQVector]:
    """
    Compute a basis for the null space of a matrix.

    The null space consists of all vectors v such that matrix * v = 0.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return []

    arr = to_numpy_matrix(matrix)

    # Compute null space using SVD
    U, s, Vt = np.linalg.svd(arr, full_matrices=True)

    # Find singular values that are essentially zero
    tolerance = 1e-10
    rank = np.sum(s > tolerance)

    # The null space is spanned by the last (n - rank) columns of V
    null_vecs = []
    for i in range(rank, Vt.shape[0]):
        vec_entries = {}
        for j in range(Vt.shape[1]):
            val = Vt[i, j]
            if abs(val) > tolerance:
                vec_entries[j] = Fraction(val).limit_denominator(10000)

        if vec_entries:  # Only add non-zero vectors
            null_vecs.append(QQVector(vec_entries))

    return null_vecs


def column_space(matrix: QQMatrix) -> List[QQVector]:
    """
    Compute a basis for the column space of a matrix.

    The column space is the span of the columns of the matrix.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not matrix.rows:
        return []

    arr = to_numpy_matrix(matrix)

    # Compute column space using QR decomposition
    # The first r columns of Q form a basis for the column space
    # where r is the rank

    Q, R = np.linalg.qr(arr)

    # Determine rank by checking R diagonal
    tolerance = 1e-10
    rank = 0
    for i in range(min(R.shape)):
        if abs(R[i, i]) > tolerance:
            rank += 1

    # Extract basis vectors
    basis = []
    for i in range(rank):
        vec_entries = {}
        for j in range(Q.shape[0]):
            val = Q[j, i]
            if abs(val) > tolerance:
                vec_entries[j] = Fraction(val).limit_denominator(10000)

        if vec_entries:
            basis.append(QQVector(vec_entries))

    return basis


def gram_schmidt(vectors: List[QQVector]) -> List[QQVector]:
    """
    Apply Gram-Schmidt orthogonalization to a list of vectors.

    Returns an orthogonal basis for the span of the input vectors.
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for this operation")

    if not vectors:
        return []

    # Convert to numpy
    # Get all dimensions
    all_dims = set()
    for vec in vectors:
        all_dims.update(vec.dimensions())

    if not all_dims:
        return []

    max_dim = max(all_dims) + 1

    # Build matrix with vectors as columns
    arr = np.zeros((max_dim, len(vectors)))
    for j, vec in enumerate(vectors):
        for i in range(max_dim):
            arr[i, j] = float(vec.get(i, QQ(0)))

    # Apply Gram-Schmidt
    Q, _ = np.linalg.qr(arr)

    # Convert back to QQVectors
    result = []
    for j in range(Q.shape[1]):
        vec_entries = {}
        for i in range(Q.shape[0]):
            val = Q[i, j]
            if abs(val) > 1e-10:
                vec_entries[i] = Fraction(val).limit_denominator(10000)

        if vec_entries:
            result.append(QQVector(vec_entries))

    return result


# Export functions
__all__ = [
    'to_numpy_matrix', 'from_numpy_matrix', 'rational_eigenvalues', 'eigenvectors',
    'matrix_power', 'determinant', 'matrix_inverse', 'qr_decomposition', 'svd',
    'null_space', 'column_space', 'gram_schmidt'
]
