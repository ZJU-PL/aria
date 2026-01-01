"""
Tests for the Linear Transition System (LTS) module.
"""

import unittest
from fractions import Fraction
from aria.srk.lts import PartialLinearMap, LTSOperations
from aria.srk.linear import QQMatrix, QQVector, mk_matrix, mk_vector, identity_matrix


class TestPartialLinearMap(unittest.TestCase):
    """Test Partial Linear Map operations."""

    def test_compose1(self):
        """Test composition of partial linear maps."""
        # Create compatible matrices for composition
        # Matrix A (2x3) * Matrix B (3x2) = Matrix C (2x2)
        m1 = mk_matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        guard1 = []
        f = PartialLinearMap.make(m1, guard1)

        m2 = mk_matrix([[1, 2], [3, 4], [5, 6]])  # 3x2
        guard2 = []
        ff = PartialLinearMap.make(m2, guard2)

        result = f.compose(ff)  # Note: order matters for composition

        # Expected result: [[22, 28], [49, 64]]
        expected_m = mk_matrix([[22, 28], [49, 64]])
        expected = PartialLinearMap.make(expected_m, [])

        self.assertEqual(result, expected)

    def test_compose2(self):
        """Test another composition case."""
        # Simple test with 2x2 matrices
        m1 = mk_matrix([[0, 1], [1, 0]])
        guard1 = []
        f = PartialLinearMap.make(m1, guard1)

        m2 = mk_matrix([[1, 0], [0, 1]])
        guard2 = []
        ff = PartialLinearMap.make(m2, guard2)

        result = f.compose(f)
        # Expected result: [[0, 1]; [1, 0]] ∘ [[0, 1]; [1, 0]] = [[1, 0]; [0, 1]]
        expected_m = mk_matrix([[1, 0], [0, 1]])
        expected = PartialLinearMap.make(expected_m, [])

        self.assertEqual(result, expected)


class TestLTS(unittest.TestCase):
    """Test Linear Transition System operations."""

    def test_determinize1(self):
        """Test LTS determinization case 1."""
        a = mk_matrix([[1, 0], [0, 0]])
        b = mk_matrix([[1, 1], [0, 1]])

        dlts, sim = LTSOperations.determinize((a, b))

        # Expected similarity matrix
        sim_expected = mk_matrix([[1, 0], [0, 1]])

        # Expected deterministic LTS
        dlts_expected = PartialLinearMap.make(mk_matrix([[1, 0], [0, 0]]), [])

        self.assertEqual(sim, sim_expected)
        self.assertEqual(dlts, dlts_expected)

    def test_determinize2(self):
        """Test LTS determinization case 2."""
        a = mk_matrix([[1, 0], [0, 1], [0, 0]])
        b = mk_matrix([[1, 1], [1, 0], [0, 1]])

        dlts, sim = LTSOperations.determinize((a, b))

        # Expected similarity matrix
        sim_expected = mk_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Expected deterministic LTS
        dlts_expected = PartialLinearMap.make(
            mk_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), []
        )

        self.assertEqual(sim, sim_expected)
        self.assertEqual(dlts, dlts_expected)

    def test_determinize3(self):
        """Test LTS determinization case 3."""
        a = mk_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        b = mk_matrix([[2, 1, 0], [3, 3, 1], [1, 0, 0]])

        dlts, sim = LTSOperations.determinize((a, b))

        # Expected deterministic LTS
        dlts_expected = PartialLinearMap.make(
            mk_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), []
        )

        # Check that the LTS is equivalent up to similarity
        id_matrix = identity_matrix(3)
        self.assertEqual(
            LTSOperations.dlts_inverse_image(id_matrix, dlts_expected),
            LTSOperations.dlts_inverse_image(sim, dlts),
        )


if __name__ == "__main__":
    unittest.main()
