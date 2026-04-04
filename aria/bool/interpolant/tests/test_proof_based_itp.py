"""Tests for proof-based Boolean interpolation."""

import unittest

import z3
from pysat.formula import CNF

from aria.bool.interpolant.proof_based_itp import BooleanInterpolant
from aria.utils.z3.expr import get_variables
from aria.utils.z3.solver import is_entail, is_equiv, is_sat


class TestProofBasedInterpolant(unittest.TestCase):
    """Regression tests for resolution proofs and McMillan interpolation."""

    def assert_interpolant(
        self,
        fml_a: z3.ExprRef,
        fml_b: z3.ExprRef,
        interpolant: z3.ExprRef,
        shared_vars,
    ) -> None:
        """Check the standard interpolant side conditions."""
        self.assertTrue(is_entail(fml_a, interpolant))
        self.assertFalse(is_sat(z3.And(interpolant, fml_b)))

        interpolant_vars = {str(var) for var in get_variables(interpolant)}
        expected_shared_vars = {str(var) for var in shared_vars}
        self.assertTrue(interpolant_vars.issubset(expected_shared_vars))

    def test_build_proof_from_cnf(self) -> None:
        """The resolution engine should derive and validate an empty clause."""
        p, q = z3.Bools("p q")
        cnf_a = CNF(from_clauses=[[1, 2], [-1]])
        cnf_b = CNF(from_clauses=[[-2]])

        proof = BooleanInterpolant.build_proof_from_cnf(
            cnf_a, cnf_b, var_to_expr={1: p, 2: q}
        )
        interpolant = proof.extract_interpolant()

        self.assertEqual(proof.root_clause, ())
        self.assertTrue(proof.validate())
        self.assertTrue(is_equiv(interpolant, q))

    def test_compute_itp_eliminates_a_local_variable(self) -> None:
        """Resolution on A-local pivots should eliminate local symbols."""
        p, q = z3.Bools("p q")
        fml_a = z3.And(z3.Or(p, q), z3.Not(p))
        fml_b = z3.Not(q)

        interpolant = BooleanInterpolant.compute_itp(fml_a, fml_b)

        self.assertTrue(is_equiv(interpolant, q))
        self.assert_interpolant(fml_a, fml_b, interpolant, [q])

    def test_compute_itp_for_non_cnf_formula(self) -> None:
        """Clausification should handle basic non-CNF Boolean structure."""
        x, y = z3.Bools("x y")
        fml_a = z3.And(z3.Implies(x, y), x)
        fml_b = z3.Not(y)

        proof = BooleanInterpolant.build_proof(fml_a, fml_b)
        interpolant = proof.extract_interpolant()

        self.assertTrue(proof.validate())
        self.assertTrue(is_equiv(interpolant, y))
        self.assert_interpolant(fml_a, fml_b, interpolant, [y])

    def test_satisfiable_pair_raises(self) -> None:
        """Interpolation requires an unsatisfiable partition pair."""
        x, y = z3.Bools("x y")

        with self.assertRaises(ValueError):
            BooleanInterpolant.build_proof(x, y)

    def test_curated_boolean_formula_family(self) -> None:
        """Check the Craig conditions across a curated operator corpus."""
        a, b, c = z3.Bools("a b c")
        cases = [
            (z3.BoolVal(True), z3.BoolVal(False)),
            (z3.BoolVal(False), c),
            (a, z3.Not(a)),
            (z3.And(z3.Or(a, b), z3.Not(a)), z3.Not(b)),
            (z3.And(z3.Implies(a, b), a), z3.Not(b)),
            (z3.Xor(a, b), z3.And(a, b)),
            (z3.Distinct(a, b), z3.And(a, b)),
            (z3.And(z3.If(a, b, z3.Not(b)), a), z3.Not(b)),
        ]

        for fml_a, fml_b in cases:
            proof = BooleanInterpolant.build_proof(fml_a, fml_b)
            interpolant = proof.extract_interpolant()
            b_vars = {str(var) for var in get_variables(fml_b)}
            shared_vars = [
                var for var in get_variables(fml_a) if str(var) in b_vars
            ]

            self.assertTrue(proof.validate())
            self.assert_interpolant(fml_a, fml_b, interpolant, shared_vars)


if __name__ == "__main__":
    unittest.main()
