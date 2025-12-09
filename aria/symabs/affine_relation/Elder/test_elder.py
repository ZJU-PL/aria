"""Tests for the Elder abstract domains implementation."""

import unittest
import numpy as np
import z3
from .matrix_ops import Matrix
from .mos_domain import MOS, alpha_mos, create_z3_variables
from .ks_domain import KS
from .ag_domain import AG
from .conversions import mos_to_ks, ks_to_mos, ag_to_ks, ks_to_ag, ag_to_mos


class TestMatrixOps(unittest.TestCase):
    """Test matrix operations and Howell form algorithms."""

    def test_matrix_creation(self):
        """Test basic matrix creation."""
        data = np.array([[1, 2], [3, 4]], dtype=object)
        matrix = Matrix(data, 8)
        self.assertEqual(matrix.rows, 2)
        self.assertEqual(matrix.cols, 2)
        self.assertEqual(matrix.modulus, 8)

    def test_matrix_copy(self):
        """Test matrix copying."""
        data = np.array([[1, 2]], dtype=object)
        matrix = Matrix(data, 16)
        copied = matrix.copy()
        self.assertEqual(matrix.rows, copied.rows)
        self.assertEqual(matrix.cols, copied.cols)
        self.assertEqual(matrix.modulus, copied.modulus)


class TestMOS(unittest.TestCase):
    """Test MOS domain implementation."""

    def test_empty_mos(self):
        """Test empty MOS element."""
        mos = MOS(w=32)
        self.assertTrue(mos.is_empty())
        self.assertEqual(mos.concretize(), "∅")

    def test_identity_mos(self):
        """Test identity transformation in MOS."""
        k = 2
        identity_data = np.eye(k + 1, dtype=object)
        identity_matrix = Matrix(identity_data, 64)
        mos = MOS([identity_matrix], 32)

        self.assertFalse(mos.is_empty())
        self.assertEqual(len(mos.matrices), 1)

    def test_mos_union(self):
        """Test MOS union operation."""
        # Create two MOS elements
        k = 1
        data1 = np.array([[1, 0], [0, 1]], dtype=object)
        data2 = np.array([[2, 0], [0, 1]], dtype=object)

        mos1 = MOS([Matrix(data1, 32)], 32)
        mos2 = MOS([Matrix(data2, 32)], 32)

        union = mos1.union(mos2)
        self.assertEqual(len(union.matrices), 2)


class TestKS(unittest.TestCase):
    """Test KS domain implementation."""

    def test_empty_ks(self):
        """Test empty KS element."""
        ks = KS(w=32)
        self.assertTrue(ks.is_empty())
        self.assertEqual(ks.concretize(), "∅")

    def test_ks_creation(self):
        """Test KS element creation."""
        k = 2
        data = np.zeros((1, 2*k + 1), dtype=object)
        data[0, 0] = 1  # x coefficient
        data[0, 2] = 1  # y' coefficient
        data[0, 4] = 0  # Constant

        ks = KS(Matrix(data, 32), 32)
        self.assertFalse(ks.is_empty())
        self.assertEqual(ks.rows, 1)
        self.assertEqual(ks.cols, 5)

    def test_ks_join(self):
        """Test KS join operation."""
        k = 1
        data1 = np.zeros((1, 3), dtype=object)
        data1[0, 0] = 1
        data1[0, 1] = -1

        data2 = np.zeros((1, 3), dtype=object)
        data2[0, 0] = 2
        data2[0, 1] = -2

        ks1 = KS(Matrix(data1, 32), 32)
        ks2 = KS(Matrix(data2, 32), 32)

        joined = ks1.join(ks2)
        self.assertEqual(joined.rows, 2)


class TestAG(unittest.TestCase):
    """Test AG domain implementation."""

    def test_empty_ag(self):
        """Test empty AG element."""
        ag = AG(w=32)
        self.assertTrue(ag.is_empty())
        self.assertEqual(ag.concretize(), "∅")

    def test_ag_creation(self):
        """Test AG element creation."""
        k = 1
        data = np.zeros((1, 3), dtype=object)
        data[0, 0] = 1  # x coefficient
        data[0, 1] = 1  # x' coefficient

        ag = AG(Matrix(data, 32), 32)
        self.assertFalse(ag.is_empty())
        self.assertEqual(ag.rows, 1)
        self.assertEqual(ag.cols, 3)


class TestConversions(unittest.TestCase):
    """Test conversion algorithms between domains."""

    def test_identity_conversion_cycle(self):
        """Test that conversions preserve identity relations."""
        k = 2
        w = 32

        # Start with identity in MOS
        identity_data = np.eye(k + 1, dtype=object)
        identity_matrix = Matrix(identity_data, 2**w)
        mos_original = MOS([identity_matrix], w)

        # Convert MOS -> KS -> AG -> MOS
        ks_version = mos_to_ks(mos_original)
        ag_version = ks_to_ag(ks_version)
        mos_final = ag_to_mos(ag_version)

        # Should have same number of matrices (for this simple case)
        self.assertEqual(len(mos_original.matrices), len(mos_final.matrices))

    def test_simple_transformation(self):
        """Test conversion of simple affine transformation."""
        k = 1
        w = 32

        # Transformation: x' = x + 1
        transform_data = np.array([[1, 1], [0, 1]], dtype=object)
        transform_matrix = Matrix(transform_data, 2**w)
        mos_transform = MOS([transform_matrix], w)

        # Should convert without error
        ks_transform = mos_to_ks(mos_transform)
        ag_transform = ks_to_ag(ks_transform)
        mos_final = ag_to_mos(ag_transform)

        self.assertFalse(mos_final.is_empty())


class TestAlphaFunction(unittest.TestCase):
    """Test the alpha function implementation."""

    def test_alpha_mos_simple(self):
        """Test alpha function on simple formula."""
        variables = ['x']
        pre_vars, post_vars = create_z3_variables(variables)
        phi = post_vars[0] == pre_vars[0]  # x' = x

        result = alpha_mos(phi, pre_vars, post_vars)
        # Should produce some abstraction (even if simplified)
        self.assertIsInstance(result, MOS)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
