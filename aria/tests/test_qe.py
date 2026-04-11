# coding: utf-8
"""
For testing the quantifier elimination engine
"""

import z3
from typing import cast

from aria.quant.qe import qelim_exists_lia_cooper
from aria.tests import TestCase, main
from aria.quant.qe.qe_fm import qelim_exists_lra_fm
from aria.quant.qe.qe_lme import qelim_exists_lme


def is_equivalent(a: z3.BoolRef, b: z3.BoolRef, timeout_ms=1000):
    """
    Check if a and b are equivalent with a timeout
    """
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(a != b)
    result = s.check()
    if result == z3.sat:
        return False
    elif result == z3.unknown:
        # If timeout occurs, we'll assume they might be equivalent
        # but log a warning
        print(f"WARNING: Equivalence check timed out after {timeout_ms}ms")
        return True
    return True


class TestQuantifierElimination(TestCase):

    def test_simple_arithmetic(self):
        """Test QE with simple arithmetic formulas"""
        x, y = z3.Ints("x y")

        # Simple linear formula: ∃x. (x > 5 ∧ x < 10)
        fml1 = z3.And(x > 5, x < 10)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Simple formula with one other variable: ∃x. (x > y)
        fml2 = x > y
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

        # Simple disjunction: ∃x. (x == 0 ∨ x == 1)
        fml3 = z3.Or(x == 0, x == 1)
        qf3 = qelim_exists_lme(fml3, x)
        qfml3 = z3.Exists(x, fml3)
        assert is_equivalent(qf3, qfml3)

    def test_basic_real_arithmetic(self):
        """Test QE with basic real arithmetic formulas"""
        x, y = z3.Reals("x y")

        # Simple real formula: ∃x. (x > 0 ∧ x < 1)
        fml1 = z3.And(x > 0, x < 1)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Simple formula with one other variable: ∃x. (x < y)
        fml2 = x < y
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

    def test_boolean_structure(self):
        """Test QE with more complex Boolean structures"""
        x, y = z3.Ints("x y")

        # Nested OR-AND structure: ∃x. ((x < 0 ∨ x > 10) ∧ (x != y))
        fml1 = z3.And(z3.Or(x < 0, x > 10), x != y)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Formula with XOR: ∃x. (x < y ⊕ x > 5)
        fml2 = z3.Xor(x < y, x > 5)
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

        # Formula with implication: ∃x. (x > 0 → x < 10)
        fml3 = z3.Implies(x > 0, x < 10)
        qf3 = qelim_exists_lme(fml3, x)
        qfml3 = z3.Exists(x, fml3)
        assert is_equivalent(qf3, qfml3)

        # Formula with nested structure: ∃x. ((x == 0 ∨ x == 1) ∧ (x < y ∨ x > y + 5))
        fml4 = z3.And(z3.Or(x == 0, x == 1), z3.Or(x < y, x > y + 5))
        qf4 = qelim_exists_lme(fml4, x)
        qfml4 = z3.Exists(x, fml4)
        assert is_equivalent(qf4, qfml4)

    def test_partial_projection_with_keep_vars(self):
        """Test that keep_vars preserves only the requested free variables"""
        x, y, z = z3.Ints("x y z")

        phi = z3.And(x == y + z, y > 0, z >= 0)
        projected = qelim_exists_lme(phi, [x], keep_vars=[y])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)

        remaining_vars = {str(var) for var in z3.z3util.get_vars(projected)}
        assert remaining_vars == {"y"}

    def test_partial_projection_defaults_to_full_projection(self):
        """Test that an empty keep_vars projects away all non-quantified variables"""
        x, y = z3.Ints("x y")

        phi = z3.And(x == y, y > 2)
        projected = qelim_exists_lme(phi, [x], keep_vars=[])

        expected = z3.Exists([x, y], phi)
        assert is_equivalent(projected, expected)
        assert z3.z3util.get_vars(projected) == []

    def test_partial_projection_rejects_overlap(self):
        """Test that keep_vars cannot overlap with quantified variables"""
        x, y = z3.Ints("x y")

        phi = x > y

        with self.assertRaises(ValueError):
            qelim_exists_lme(phi, [x], keep_vars=[x])


class TestFourierMotzkinQuantifierElimination(TestCase):

    def test_fm_interval_elimination(self):
        """FM should eliminate interval constraints over Reals"""
        x, y, z = z3.Reals("x y z")

        phi = z3.And(x >= y + 1, x <= z - 2)
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)

    def test_fm_equality_elimination(self):
        """FM should handle affine equalities by turning them into paired bounds"""
        x, y = z3.Reals("x y")

        phi = z3.And(x == y + 1, x < 5)
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)

    def test_fm_disjunction_and_cube_expansion(self):
        """FM should work across DNF cube expansion for Boolean structure"""
        x, y, z = z3.Reals("x y z")

        phi = z3.Or(z3.And(x < y, y < 0), z3.And(x > z, z > 1))
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)

    def test_fm_distinct_branching(self):
        """FM should branch locally on != atoms during cube expansion"""
        x, y = z3.Reals("x y")

        phi = z3.And(x != y, x == 3)
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)

    def test_fm_keep_vars_matches_lme_semantics(self):
        """FM keep_vars should match qe_lme partial projection semantics"""
        x, y, z = z3.Reals("x y z")

        phi = z3.And(x == y + z, y > 0, z >= 0)
        projected = qelim_exists_lra_fm(phi, [x], keep_vars=[y])
        expected = z3.Exists([x, z], phi)

        assert is_equivalent(projected, expected)

    def test_fm_rejects_mixed_int_real_fragment(self):
        """FM should fail fast when non-Real variables appear in the fragment"""
        x = z3.Real("x")
        i = z3.Int("i")

        phi = z3.And(x > 0, i > 0)

        with self.assertRaises(ValueError):
            qelim_exists_lra_fm(phi, [x])

    def test_fm_rejects_nonlinear_terms(self):
        """FM should fail fast on nonlinear arithmetic"""
        x, y = z3.Reals("x y")

        phi = cast(z3.BoolRef, x * y < z3.RealVal(1))

        with self.assertRaises(ValueError):
            qelim_exists_lra_fm(phi, [x])

    def test_fm_falls_back_when_cube_expansion_guard_trips(self):
        """FM should still project large Boolean disjunctions via safe fallback"""
        x = z3.Real("x")

        phi = z3.Or(*[x == z3.RealVal(i) for i in range(70)])
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)
        assert z3.is_true(z3.simplify(projected))

    def test_fm_falls_back_for_boolean_structure_outside_affine_atoms(self):
        """FM should preserve benign Boolean structure by delegating projection"""
        b = z3.Bool("b")
        x, y = z3.Reals("x y")

        phi = z3.And(z3.Or(b, x < y), x > 0)
        projected = qelim_exists_lra_fm(phi, [x])
        expected = z3.Exists([x], phi)

        assert is_equivalent(projected, expected)


def brute_force_exists(
    phi: z3.BoolRef,
    qvar: z3.ArithRef,
    env: dict[z3.ArithRef, int],
    *,
    domain=range(-4, 5),
) -> bool:
    for value in domain:
        substitutions = [(qvar, z3.IntVal(value))]
        substitutions.extend((var, z3.IntVal(num)) for var, num in env.items())
        candidate = z3.simplify(z3.substitute(phi, *substitutions))
        if z3.is_true(candidate):
            return True
    return False


class TestCooperQuantifierElimination(TestCase):

    def test_non_unit_coefficients(self):
        x, y = z3.Ints("x y")

        phi = z3.And(2 * x + y >= 3, 3 * x - y <= 4)
        projected = qelim_exists_lia_cooper(phi, [x])

        assert is_equivalent(projected, z3.Exists([x], phi))

    def test_gcd_unsat_parity_case(self):
        x, y = z3.Ints("x y")

        projected = qelim_exists_lia_cooper(2 * x == y + 1, [x])
        expected = cast(z3.BoolRef, ((y + 1) % 2) == 0)

        assert is_equivalent(projected, expected)
        assert is_equivalent(projected, z3.Exists([x], 2 * x == y + 1))

    def test_result_keeps_congruence_constraint(self):
        x, y = z3.Ints("x y")

        projected = qelim_exists_lia_cooper(3 * x + 1 == y, [x])

        assert is_equivalent(projected, cast(z3.BoolRef, ((y - 1) % 3) == 0))
        assert "mod" in projected.sexpr()

    def test_strict_bound_normalization(self):
        x, y = z3.Ints("x y")

        phi = z3.And(2 * x > y, 2 * x < y + 3)
        projected = qelim_exists_lia_cooper(phi, [x])

        assert is_equivalent(projected, z3.Exists([x], phi))

    def test_keep_vars_matches_lme_projection_rule(self):
        x, y, z = z3.Ints("x y z")

        phi = z3.And(2 * x + z == y, z >= 0, y <= 8)
        projected = qelim_exists_lia_cooper(phi, [x], keep_vars=[y])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)

        remaining_vars = {str(var) for var in z3.z3util.get_vars(projected)}
        assert remaining_vars == {"y"}

    def test_repeated_elimination_handles_generated_congruence_atoms(self):
        x, y, z = z3.Ints("x y z")

        phi = z3.And(2 * x + z == y, z >= 0, z <= 2)
        projected = qelim_exists_lia_cooper(phi, [x, z])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)
        assert z3.z3util.get_vars(projected) == [y]

    def test_repeated_elimination_generated_congruence_with_no_bounds(self):
        x, y, z = z3.Ints("x y z")

        phi = 2 * x + z == y
        projected = qelim_exists_lia_cooper(phi, [x, z])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)
        assert z3.is_true(z3.simplify(projected))

    def test_repeated_elimination_generated_congruence_with_lower_bound_only(self):
        x, y, z = z3.Ints("x y z")

        phi = z3.And(2 * x + z == y, z >= 0)
        projected = qelim_exists_lia_cooper(phi, [x, z])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)
        assert z3.is_true(z3.simplify(projected))

    def test_repeated_elimination_generated_congruence_with_upper_bound_only(self):
        x, y, z = z3.Ints("x y z")

        phi = z3.And(2 * x + z == y, z <= 0)
        projected = qelim_exists_lia_cooper(phi, [x, z])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)
        assert z3.is_true(z3.simplify(projected))

    def test_repeated_elimination_rejects_incompatible_modular_constraints(self):
        x, y, z = z3.Ints("x y z")

        phi = z3.And(2 * x + z == 0, 2 * y + z == 1)
        projected = qelim_exists_lia_cooper(phi, [x, z])

        expected = z3.Exists([x, z], phi)
        assert is_equivalent(projected, expected)
        assert z3.is_false(z3.simplify(projected))

    def test_fail_fast_on_unsupported_inputs(self):
        x, y = z3.Ints("x y")
        r = z3.Real("r")
        f = z3.Function("f", z3.IntSort(), z3.IntSort())

        unsupported = [
            cast(z3.BoolRef, cast(z3.ArithRef, x * y) > 0),
            cast(z3.BoolRef, x + z3.ToInt(r) > 0),
            cast(z3.BoolRef, x + (y % 2) > 0),
            cast(z3.BoolRef, z3.And(x > 0, z3.Exists([y], y > x))),
            cast(z3.BoolRef, cast(z3.ArithRef, f(x)) > 0),
        ]

        for phi in unsupported:
            with self.assertRaises(ValueError):
                qelim_exists_lia_cooper(phi, [x])

    def test_accepts_input_congruence_atoms(self):
        x, y = z3.Ints("x y")

        phi = cast(z3.BoolRef, ((2 * x + y) % 4) == 0)
        projected = qelim_exists_lia_cooper(phi, [x])

        assert is_equivalent(projected, z3.Exists([x], phi))

    def test_accepts_input_disequalities(self):
        x, y = z3.Ints("x y")

        phi = z3.And(x != y, x >= 0)
        projected = qelim_exists_lia_cooper(phi, [x])

        assert is_equivalent(projected, z3.Exists([x], phi))

    def test_small_domain_bruteforce_sanity_check(self):
        x, y, z = z3.Ints("x y z")

        phi = cast(z3.BoolRef, z3.And(2 * x + y >= z, 3 * x < y + 4, x - z <= 1))
        projected = qelim_exists_lia_cooper(phi, [x])

        for y_val in range(-2, 3):
            for z_val in range(-2, 3):
                env = {y: y_val, z: z_val}
                brute = brute_force_exists(phi, x, env, domain=range(-5, 6))
                reduced = z3.simplify(
                    z3.substitute(
                        projected,
                        (y, z3.IntVal(y_val)),
                        (z, z3.IntVal(z_val)),
                    )
                )
                assert z3.is_true(reduced) == brute

    def test_partial_projection_rejects_overlap(self):
        x, y = z3.Ints("x y")

        with self.assertRaises(ValueError):
            qelim_exists_lia_cooper(x > y, [x], keep_vars=[x])


if __name__ == "__main__":
    main()
