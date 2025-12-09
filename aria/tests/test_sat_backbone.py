"""
Tests for SAT backbone algorithms
"""

import unittest
from pysat.formula import CNF
from aria.backbone import (
    compute_backbone,
    compute_backbone_iterative,
    compute_backbone_chunking,
    compute_backbone_refinement,
    compute_backbone_with_approximation,
    is_backbone_literal,
    BackboneAlgorithm
)


class TestSATBackbone(unittest.TestCase):
    """Test cases for SAT backbone computation algorithms"""

    def setUp(self):
        # Create a simple CNF formula: (a ∨ b) ∧ (¬b ∨ c) ∧ (¬a ∨ c)
        # Variables are 1=a, 2=b, 3=c
        # The only backbone literal is c (3)
        self.cnf1 = CNF()
        self.cnf1.append([1, 2])  # a ∨ b
        self.cnf1.append([-2, 3])  # ¬b ∨ c
        self.cnf1.append([-1, 3])  # ¬a ∨ c

        # Create another CNF formula: (a) ∧ (b ∨ c) ∧ (¬b ∨ d)
        # Variables are 1=a, 2=b, 3=c, 4=d
        # Backbone literals are a (1) and (b → d), i.e., (¬b ∨ d) which is not a single literal
        self.cnf2 = CNF()
        self.cnf2.append([1])  # a
        self.cnf2.append([2, 3])  # b ∨ c
        self.cnf2.append([-2, 4])  # ¬b ∨ d

        # Unsatisfiable formula: a ∧ ¬a
        self.unsat_cnf = CNF()
        self.unsat_cnf.append([1])  # a
        self.unsat_cnf.append([-1])  # ¬a

        # Formula with all literals in backbone: a ∧ b ∧ c
        self.all_backbone_cnf = CNF()
        self.all_backbone_cnf.append([1])  # a
        self.all_backbone_cnf.append([2])  # b
        self.all_backbone_cnf.append([3])  # c

    def test_backbone_iterative(self):
        backbone, calls = compute_backbone_iterative(self.cnf1)
        self.assertEqual(set(backbone), {3})

        backbone, calls = compute_backbone_iterative(self.cnf2)
        self.assertEqual(set(backbone), {1})

        backbone, calls = compute_backbone_iterative(self.unsat_cnf)
        self.assertEqual(backbone, [])

        backbone, calls = compute_backbone_iterative(self.all_backbone_cnf)
        self.assertEqual(set(backbone), {1, 2, 3})

    def test_backbone_chunking(self):
        backbone, calls = compute_backbone_chunking(self.cnf1)
        self.assertEqual(set(backbone), {3})

        backbone, calls = compute_backbone_chunking(self.cnf2)
        self.assertEqual(set(backbone), {1})

        backbone, calls = compute_backbone_chunking(self.unsat_cnf)
        self.assertEqual(backbone, [])

        backbone, calls = compute_backbone_chunking(self.all_backbone_cnf)
        self.assertEqual(set(backbone), {1, 2, 3})

    def test_backbone_refinement(self):
        backbone, calls = compute_backbone_refinement(self.cnf1)
        self.assertEqual(set(backbone), {3})

        backbone, calls = compute_backbone_refinement(self.cnf2)
        self.assertEqual(set(backbone), {1})

        backbone, calls = compute_backbone_refinement(self.unsat_cnf)
        self.assertEqual(backbone, [])

        backbone, calls = compute_backbone_refinement(self.all_backbone_cnf)
        self.assertEqual(set(backbone), {1, 2, 3})

    def test_backbone_default(self):
        backbone, calls = compute_backbone(self.cnf1)
        self.assertEqual(set(backbone), {3})

        backbone, calls = compute_backbone(self.cnf2)
        self.assertEqual(set(backbone), {1})

    def test_is_backbone_literal(self):
        is_backbone, calls = is_backbone_literal(self.cnf1, 3)
        self.assertTrue(is_backbone)

        is_backbone, calls = is_backbone_literal(self.cnf1, 1)
        self.assertFalse(is_backbone)

        is_backbone, calls = is_backbone_literal(self.cnf2, 1)
        self.assertTrue(is_backbone)

        is_backbone, calls = is_backbone_literal(self.unsat_cnf, 1)
        self.assertFalse(is_backbone)

    def test_backbone_approximation(self):
        definite, potential, calls = compute_backbone_with_approximation(self.cnf1)
        self.assertIn(3, definite)
        self.assertIn(3, potential)

        definite, potential, calls = compute_backbone_with_approximation(self.cnf2)
        self.assertIn(1, definite)
        self.assertIn(1, potential)

        definite, potential, calls = compute_backbone_with_approximation(self.unsat_cnf)
        self.assertEqual(definite, [])
        self.assertEqual(potential, [])

        definite, potential, calls = compute_backbone_with_approximation(self.all_backbone_cnf)
        self.assertEqual(set(definite), {1, 2, 3})
        self.assertEqual(set(potential), {1, 2, 3})


if __name__ == '__main__':
    unittest.main()
