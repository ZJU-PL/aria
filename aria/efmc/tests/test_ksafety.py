"""
Unit tests for k-safety verification engine.
"""

import pytest
import unittest
import z3
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ksafety import (
    BaseKSafetyProver,
    NonInterferenceProver,
    DeterminismProver,
    SymmetryProver,
    DifferentialPrivacyProver,
    EquivalenceProver,
    RefinementProver,
    HyperLTLProver,
)
from aria.efmc.engines.ksafety.hyperltl import Var, Atom, G, Implies
from aria.efmc.utils.verification_utils import VerificationResult


class TestBaseKSafetyProver(unittest.TestCase):
    """Test cases for base k-safety verification functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple transition system for testing
        self.x = z3.Int("x")
        self.y = z3.Int("y")
        self.secret = z3.Int("secret")
        self.output = z3.Int("output")

        # Create transition system
        self.sts = TransitionSystem()
        self.sts.variables = [self.x, self.y, self.secret, self.output]
        self.sts.prime_variables = [
            z3.Int("x_p"),
            z3.Int("y_p"),
            z3.Int("secret_p"),
            z3.Int("output_p"),
        ]
        self.sts.init = z3.And(self.x == 0, self.y == 0, self.output == 0)
        self.sts.trans = z3.And(
            z3.Implies(
                self.secret > 0,
                z3.And(self.x == self.x + 1, self.y == self.y + self.secret),
            ),
            z3.Implies(
                self.secret <= 0, z3.And(self.x == self.x + 1, self.y == self.y)
            ),
            self.output == self.x + self.y,
        )
        self.sts.post = z3.And(self.x >= 0, self.y >= 0)

    def test_initialization(self):
        """Test k-safety prover initialization."""
        prover = BaseKSafetyProver(self.sts, k=2)

        self.assertEqual(prover.k, 2)
        self.assertEqual(prover.sts, self.sts)
        self.assertEqual(prover.timeout, 300)
        self.assertEqual(prover.verification_method, "bounded_model_checking")

    def test_create_trace_variables(self):
        """Test trace variable creation."""
        prover = BaseKSafetyProver(self.sts, k=2)
        trace_vars = prover.create_trace_variables()

        # Check that we have 2 traces
        self.assertEqual(len(trace_vars), 2)

        # Check that each trace has the same number of variables as the original system
        for trace_idx in range(2):
            self.assertEqual(len(trace_vars[trace_idx]), len(self.sts.variables))

            # Check variable naming
            for i, var in enumerate(self.sts.variables):
                trace_var = trace_vars[trace_idx][i]
                var_name = str(var)
                expected_name = f"{var_name}_{trace_idx}"
                self.assertIn(expected_name, str(trace_var))

    def test_set_relational_property(self):
        """Test setting relational property."""
        prover = BaseKSafetyProver(self.sts, k=2)

        # Create a simple property
        trace_vars = prover.create_trace_variables()
        property_expr = trace_vars[0][0] >= 0  # x >= 0 in first trace

        prover.set_relational_property(property_expr)
        self.assertEqual(prover.relational_property, property_expr)

    def test_create_k_safety_formula(self):
        """Test k-safety formula creation."""
        prover = BaseKSafetyProver(self.sts, k=2)

        # Set a simple property
        trace_vars = prover.create_trace_variables()
        property_expr = trace_vars[0][0] >= 0
        prover.set_relational_property(property_expr)

        # Create the formula
        formula = prover.create_k_safety_formula()

        # Check that the formula is not None
        self.assertIsNotNone(formula)
        self.assertEqual(prover.verification_formula, formula)

    def test_create_k_safety_formula_without_property(self):
        """Test that creating formula without property raises error."""
        prover = BaseKSafetyProver(self.sts, k=2)

        with self.assertRaises(ValueError):
            prover.create_k_safety_formula()

    def test_bounded_model_checking(self):
        """Test BMC with different properties."""
        prover = BaseKSafetyProver(self.sts, k=2, timeout=10)

        # Test with a safe property: x >= 0 in first trace
        trace_vars = prover.create_trace_variables()
        property_expr = trace_vars[0][0] >= 0  # x >= 0
        prover.set_relational_property(property_expr)

        result = prover.bounded_model_checking(1)

        # Should be safe since x starts at 0 and only increases
        self.assertTrue(result.is_safe or result.is_unknown)

    def test_solve_without_property(self):
        """Test that solve without property raises error."""
        prover = BaseKSafetyProver(self.sts, k=2)

        with self.assertRaises(ValueError):
            prover.solve()

    def test_different_k_values(self):
        """Test with different k values."""
        for k in [2, 3]:
            prover = BaseKSafetyProver(self.sts, k=k, timeout=10)

            # Create property for k traces
            trace_vars = prover.create_trace_variables()
            property_expr = z3.And(*[trace_vars[i][0] >= 0 for i in range(k)])
            prover.set_relational_property(property_expr)

            result = prover.solve()

            # Check that we get a valid result
            self.assertIsInstance(result, VerificationResult)

    def test_verification_methods(self):
        """Test different verification methods."""
        methods = ["bounded_model_checking", "k_induction"]

        for method in methods:
            prover = BaseKSafetyProver(self.sts, k=2, timeout=10, method=method)

            # Set a simple property
            trace_vars = prover.create_trace_variables()
            property_expr = trace_vars[0][0] >= 0
            prover.set_relational_property(property_expr)

            result = prover.solve()

            # Check that we get a valid result
            self.assertIsInstance(result, VerificationResult)

    def test_custom_relational_property(self):
        """Test custom relational property verification."""
        prover = BaseKSafetyProver(self.sts, k=2, timeout=10)

        # Create trace variables
        trace_vars = prover.create_trace_variables()

        # Create a custom property: if x values are the same, then y values should be the same
        x_0 = trace_vars[0][0]  # x in trace 0
        x_1 = trace_vars[1][0]  # x in trace 1
        y_0 = trace_vars[0][1]  # y in trace 0
        y_1 = trace_vars[1][1]  # y in trace 1

        custom_property = z3.Implies(x_0 == x_1, y_0 == y_1)

        prover.set_relational_property(custom_property)
        result = prover.solve()

        # Check that we get a valid result
        self.assertIsInstance(result, VerificationResult)

    def test_k_induction_components(self):
        """Test k-induction base case and inductive step creation."""
        prover = BaseKSafetyProver(self.sts, k=2, timeout=10)

        # Set a property
        trace_vars = prover.create_trace_variables()
        property_expr = trace_vars[0][0] >= 0
        prover.set_relational_property(property_expr)

        # Create base case and inductive step
        base_case = prover._create_k_induction_base_case(1)
        inductive_step = prover._create_k_induction_inductive_step(1)

        # Check that both are not None
        self.assertIsNotNone(base_case)
        self.assertIsNotNone(inductive_step)


class TestKSafetyEngines(unittest.TestCase):
    """Concise tests for specialized ksafety engines."""

    def _mk_ts_det(self):
        # Variables: inp, out
        inp, out = z3.Int("inp"), z3.Int("out")
        inp_p, out_p = z3.Int("inp_p"), z3.Int("out_p")
        sts = TransitionSystem()
        sts.variables = [inp, out]
        sts.prime_variables = [inp_p, out_p]
        sts.init = z3.And(inp == 0, out == 0)
        sts.trans = z3.And(inp_p == inp, out_p == inp + 1)
        return sts

    def test_determinism_prover(self):
        sts = self._mk_ts_det()
        prover = DeterminismProver(sts, timeout=5, max_bound=3)
        res = prover.verify_determinism(input_vars=["inp"], output_vars=["out"])
        assert isinstance(res, VerificationResult)
        assert res.is_safe or res.is_unknown

    def test_non_interference_prover(self):
        # secret should not affect low output
        secret, low = z3.Int("secret"), z3.Int("low")
        secret_p, low_p = z3.Int("secret_p"), z3.Int("low_p")
        sts = TransitionSystem()
        sts.variables = [secret, low]
        sts.prime_variables = [secret_p, low_p]
        sts.init = z3.And(secret == 0, low == 0)
        # low evolves independently of secret
        sts.trans = z3.And(secret_p == secret, low_p == low + 1)
        prover = NonInterferenceProver(sts, timeout=5, max_bound=3)
        res = prover.verify_non_interference(high_vars=["secret"], low_vars=["low"])
        assert isinstance(res, VerificationResult)
        assert res.is_safe or res.is_unknown

    def test_symmetry_prover(self):
        # Symmetry under swapping a and b; sum is symmetric
        a, b, s = z3.Int("a"), z3.Int("b"), z3.Int("s")
        a_p, b_p, s_p = z3.Int("a_p"), z3.Int("b_p"), z3.Int("s_p")
        sts = TransitionSystem()
        sts.variables = [a, b, s]
        sts.prime_variables = [a_p, b_p, s_p]
        sts.init = z3.And(a == 0, b == 0, s == 0)
        sts.trans = z3.And(a_p == a, b_p == b, s_p == a + b)
        prover = SymmetryProver(sts, timeout=5, max_bound=2)
        res = prover.verify_input_permutation_symmetry(
            input_perm={"a": "b", "b": "a"}, output_perm={"s": "s"}
        )
        assert isinstance(res, VerificationResult)
        assert res.is_safe or res.is_unknown

    def test_differential_privacy_prover(self):
        # out tracks inp; epsilon-sensitivity should hold with epsilon >= delta
        inp, out = z3.Int("inp"), z3.Int("out")
        inp_p, out_p = z3.Int("inp_p"), z3.Int("out_p")
        sts = TransitionSystem()
        sts.variables = [inp, out]
        sts.prime_variables = [inp_p, out_p]
        sts.init = z3.And(inp == 0, out == 0)
        sts.trans = z3.And(inp_p == inp, out_p == inp)
        prover = DifferentialPrivacyProver(sts, timeout=5, max_bound=2)
        res = prover.verify_epsilon_sensitivity(
            input_vars=["inp"],
            output_vars=["out"],
            adjacency_bounds={"inp": 1.0},
            epsilon=1.0,
        )
        assert isinstance(res, VerificationResult)
        assert res.is_safe or res.is_unknown

    def test_equivalence_and_refinement_provers(self):
        # Two equivalent systems: out' = x + 1 in both (matching variable names across systems)
        x, out = z3.Int("x"), z3.Int("out")
        x_p, out_p = z3.Int("x_p"), z3.Int("out_p")
        sts_a = TransitionSystem()
        sts_a.variables = [x, out]
        sts_a.prime_variables = [x_p, out_p]
        sts_a.init = z3.And(x == 0, out == 0)
        sts_a.trans = z3.And(x_p == x + 1, out_p == x_p)

        xb, outb = z3.Int("x"), z3.Int("out")  # same names in second system
        xb_p, outb_p = z3.Int("x_p"), z3.Int("out_p")
        sts_b = TransitionSystem()
        sts_b.variables = [xb, outb]
        sts_b.prime_variables = [xb_p, outb_p]
        sts_b.init = z3.And(xb == 0, outb == 0)
        sts_b.trans = z3.And(xb_p == xb + 1, outb_p == xb_p)

        eq = EquivalenceProver(sts_a, sts_b, timeout=5, max_bound=2)
        res_eq = eq.verify_functional_equivalence(
            input_vars=["x"], output_vars=["out"], inputs_equal_all_steps=True
        )
        assert isinstance(res_eq, VerificationResult)
        assert res_eq.is_safe or res_eq.is_unknown

        ref = RefinementProver(sts_a, sts_b, timeout=5, max_bound=2)
        res_ref = ref.verify_refinement(input_vars=["x"], output_vars=["out"])
        assert isinstance(res_ref, VerificationResult)
        assert res_ref.is_safe or res_ref.is_unknown

    def test_hyperltl_prover(self):
        # If G(inp0 == inp1) then G(out0 == out1). System enforces out == inp.
        inp, out = z3.Int("inp"), z3.Int("out")
        inp_p, out_p = z3.Int("inp_p"), z3.Int("out_p")
        sts = TransitionSystem()
        sts.variables = [inp, out]
        sts.prime_variables = [inp_p, out_p]
        sts.init = z3.And(inp == 0, out == 0)
        sts.trans = z3.And(inp_p == inp, out_p == inp)
        prover = HyperLTLProver(sts, k=2, timeout=5, max_bound=2)
        phi = Implies(
            G(Atom("==", Var("inp", 0), Var("inp", 1))),
            G(Atom("==", Var("out", 0), Var("out", 1))),
        )
        prover.set_formula(phi)
        res = prover.bounded_model_checking(2)
        assert isinstance(res, VerificationResult)
        assert res.is_safe or res.is_unknown


if __name__ == "__main__":
    unittest.main()
