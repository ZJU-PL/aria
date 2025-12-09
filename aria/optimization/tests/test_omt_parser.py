"""Tests for OMTParser"""

import unittest
import z3
from aria.optimization.omt_parser import OMTParser


class TestOMTParser(unittest.TestCase):
    """Test cases for OMTParser"""

    def test_single_maximize(self):
        """Test single maximize objective"""
        parser = OMTParser()
        parser.parse_with_z3("(declare-fun x () Int)(assert (>= x 0))(maximize x)(check-sat)")
        self.assertEqual(len(parser.objectives), 1)
        self.assertIsNotNone(parser.objective)
        self.assertEqual(parser.original_directions, ["max"])

    def test_single_minimize(self):
        """Test single minimize objective"""
        parser = OMTParser()
        parser.parse_with_z3("(declare-fun x () Int)(assert (>= x 0))(minimize x)(check-sat)")
        self.assertEqual(len(parser.objectives), 1)
        self.assertEqual(parser.original_directions, ["min"])

    def test_multi_objective_boxed(self):
        """Test multi-objective boxed mode (independent objectives)"""
        parser = OMTParser()
        smt = "(declare-fun x () Int)(declare-fun y () Int)(assert (and (>= x 0)(>= y 0)))(maximize x)(minimize y)(check-sat)"
        parser.parse_with_z3(smt)
        self.assertEqual(len(parser.objectives), 2)
        self.assertEqual(parser.original_directions, ["max", "min"])
        self.assertIsNone(parser.objective)  # multi-obj has no single .objective

    def test_multi_objective_all_max(self):
        """Test multiple maximize objectives"""
        parser = OMTParser()
        smt = "(declare-fun x () Int)(declare-fun y () Int)(assert (>= x 0))(maximize x)(maximize y)(check-sat)"
        parser.parse_with_z3(smt)
        self.assertEqual(len(parser.objectives), 2)
        self.assertEqual(parser.original_directions, ["max", "max"])

    def test_multi_objective_all_min(self):
        """Test multiple minimize objectives"""
        parser = OMTParser()
        smt = "(declare-fun x () Int)(declare-fun y () Int)(assert (>= x 0))(minimize x)(minimize y)(check-sat)"
        parser.parse_with_z3(smt)
        self.assertEqual(len(parser.objectives), 2)
        self.assertEqual(parser.original_directions, ["min", "min"])

    def test_bitvector_objective(self):
        """Test bit-vector objectives"""
        parser = OMTParser()
        smt = "(declare-fun x () (_ BitVec 8))(assert (bvule x (_ bv100 8)))(maximize x)(check-sat)"
        parser.parse_with_z3(smt)
        self.assertEqual(len(parser.objectives), 1)
        self.assertEqual(parser.objective.sort_kind(), z3.Z3_BV_SORT)

    def test_no_objectives_error(self):
        """Test error when no objectives present"""
        parser = OMTParser()
        with self.assertRaises(ValueError):
            parser.parse_with_z3("(declare-fun x () Int)(assert (>= x 0))(check-sat)")

    def test_conflicting_normalization_error(self):
        """Test error when both normalization flags are set"""
        parser = OMTParser()
        parser.to_max_obj = parser.to_min_obj = True
        with self.assertRaises(ValueError):
            parser.parse_with_z3("(declare-fun x () Int)(maximize x)(check-sat)")


if __name__ == "__main__":
    unittest.main()
