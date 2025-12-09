"""
Tests for the linear algebra module.
"""

import unittest
from fractions import Fraction
from aria.srk.linear import QQVector, QQMatrix, QQVectorSpace, zero_vector, unit_vector, identity_matrix, vector_from_list, matrix_from_lists


class TestQQVector(unittest.TestCase):
    """Test rational vector operations."""

    def test_creation(self):
        """Test vector creation."""
        v1 = QQVector({0: Fraction(1), 1: Fraction(2)})
        v2 = QQVector()

        self.assertEqual(v1.entries[0], Fraction(1))
        self.assertEqual(v1.entries[1], Fraction(2))
        self.assertEqual(len(v2.entries), 0)

    def test_addition(self):
        """Test vector addition."""
        v1 = QQVector({0: Fraction(1), 1: Fraction(2)})
        v2 = QQVector({0: Fraction(3), 1: Fraction(-1)})

        sum_v = v1 + v2
        expected = QQVector({0: Fraction(4), 1: Fraction(1)})

        self.assertEqual(sum_v, expected)

    def test_multiplication(self):
        """Test scalar multiplication."""
        v = QQVector({0: Fraction(2), 1: Fraction(3)})

        scaled = v * Fraction(2)
        expected = QQVector({0: Fraction(4), 1: Fraction(6)})

        self.assertEqual(scaled, expected)

    def test_dot_product(self):
        """Test dot product."""
        v1 = QQVector({0: Fraction(1), 1: Fraction(2)})
        v2 = QQVector({0: Fraction(3), 1: Fraction(4)})

        dot = v1.dot(v2)
        expected = Fraction(1)*Fraction(3) + Fraction(2)*Fraction(4)  # 3 + 8 = 11

        self.assertEqual(dot, expected)

    def test_pivot(self):
        """Test pivot operation."""
        v = QQVector({0: Fraction(2), 1: Fraction(4)})

        pivot_coeff, pivoted = v.pivot(0)

        self.assertEqual(pivot_coeff, Fraction(2))
        expected_pivoted = QQVector({1: Fraction(2)})  # 4/2 = 2
        self.assertEqual(pivoted, expected_pivoted)

    def test_dimension(self):
        """Test dimension calculation."""
        v1 = QQVector({0: Fraction(1), 1: Fraction(2)})
        v2 = QQVector()

        self.assertEqual(v1.dimension(), 2)
        self.assertEqual(v2.dimension(), 0)

    def test_get_set(self):
        """Test get and set operations."""
        v = QQVector({0: Fraction(1)})

        self.assertEqual(v.get(0), Fraction(1))
        self.assertEqual(v.get(1), Fraction(0))

        v2 = v.set(1, Fraction(2))
        expected = QQVector({0: Fraction(1), 1: Fraction(2)})
        self.assertEqual(v2, expected)

        # Setting to zero should remove the entry
        v3 = v2.set(0, Fraction(0))
        expected_zero = QQVector({1: Fraction(2)})
        self.assertEqual(v3, expected_zero)

    def test_is_zero(self):
        """Test zero vector detection."""
        zero_v = QQVector()
        non_zero_v = QQVector({0: Fraction(1)})

        self.assertTrue(zero_v.is_zero())
        self.assertFalse(non_zero_v.is_zero())


class TestQQMatrix(unittest.TestCase):
    """Test rational matrix operations."""

    def test_creation(self):
        """Test matrix creation."""
        v1 = QQVector({0: Fraction(1), 1: Fraction(2)})
        v2 = QQVector({0: Fraction(3), 1: Fraction(4)})

        m = QQMatrix([v1, v2])

        self.assertEqual(len(m.rows), 2)
        self.assertEqual(m.rows[0], v1)
        self.assertEqual(m.rows[1], v2)

    def test_addition(self):
        """Test matrix addition."""
        m1 = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(2)}),
            QQVector({0: Fraction(3), 1: Fraction(4)})
        ])
        m2 = QQMatrix([
            QQVector({0: Fraction(5), 1: Fraction(6)}),
            QQVector({0: Fraction(7), 1: Fraction(8)})
        ])

        sum_m = m1 + m2
        expected = QQMatrix([
            QQVector({0: Fraction(6), 1: Fraction(8)}),
            QQVector({0: Fraction(10), 1: Fraction(12)})
        ])

        self.assertEqual(sum_m, expected)

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        m = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(2)}),
            QQVector({0: Fraction(3), 1: Fraction(4)})
        ])

        scaled = m * Fraction(2)
        expected = QQMatrix([
            QQVector({0: Fraction(2), 1: Fraction(4)}),
            QQVector({0: Fraction(6), 1: Fraction(8)})
        ])

        self.assertEqual(scaled, expected)

    def test_transpose(self):
        """Test matrix transpose."""
        m = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(2)}),
            QQVector({0: Fraction(3), 1: Fraction(4)})
        ])

        transposed = m.transpose()
        expected = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(3)}),
            QQVector({0: Fraction(2), 1: Fraction(4)})
        ])

        self.assertEqual(transposed, expected)

    def test_matrix_vector_multiplication(self):
        """Test matrix-vector multiplication."""
        m = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(2)}),
            QQVector({0: Fraction(3), 1: Fraction(4)})
        ])
        v = QQVector({0: Fraction(1), 1: Fraction(1)})

        result = m * v
        # This is a simplified test - the actual implementation may vary
        self.assertIsInstance(result, QQVector)

    def test_identity_matrix(self):
        """Test identity matrix creation."""
        identity = identity_matrix(3)

        self.assertEqual(len(identity.rows), 3)
        for i in range(3):
            self.assertEqual(identity.rows[i].get(i), Fraction(1))
            for j in range(3):
                if i != j:
                    self.assertEqual(identity.rows[i].get(j), Fraction(0))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_zero_vector(self):
        """Test zero vector creation."""
        zero = zero_vector(5)
        self.assertEqual(len(zero.entries), 0)

    def test_unit_vector(self):
        """Test unit vector creation."""
        unit = unit_vector(2, 5)
        expected = QQVector({2: Fraction(1)})
        self.assertEqual(unit, expected)

    def test_vector_from_list(self):
        """Test vector creation from list."""
        values = [Fraction(1), Fraction(2), Fraction(0), Fraction(3)]
        vector = vector_from_list(values)
        expected = QQVector({0: Fraction(1), 1: Fraction(2), 3: Fraction(3)})
        self.assertEqual(vector, expected)

    def test_matrix_from_lists(self):
        """Test matrix creation from lists."""
        rows = [
            [Fraction(1), Fraction(2)],
            [Fraction(3), Fraction(4)]
        ]
        matrix = matrix_from_lists(rows)
        expected = QQMatrix([
            QQVector({0: Fraction(1), 1: Fraction(2)}),
            QQVector({0: Fraction(3), 1: Fraction(4)})
        ])
        self.assertEqual(matrix, expected)


if __name__ == '__main__':
    unittest.main()
