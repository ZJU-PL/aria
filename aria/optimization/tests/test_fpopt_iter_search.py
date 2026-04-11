"""Tests for floating-point OMT support."""

import os
import tempfile
import unittest
from typing import cast

import z3

from aria.optimization.omt_solver import solve_opt_file
from aria.optimization.omtfp.fp_omt_parser import FPOMTParser
from aria.optimization.omtfp.fp_opt_iterative_search import (
    fp_opt_with_binary_search,
    fp_opt_with_linear_search,
)
from aria.optimization.omtfp.fp_opt_qsmt import (
    fp_opt_with_qsmt,
    fp_optimize_pareto,
    fp_total_key,
)


def _fp_bits(value: z3.ExprRef) -> int:
    if value.num_args() == 1:
        return int(str(value.arg(0)))

    probe = z3.FP("probe", value.sort())
    solver = z3.Solver()
    solver.add(z3.fpToIEEEBV(probe) == z3.fpToIEEEBV(value))
    assert solver.check() == z3.sat
    bits = cast(z3.BitVecNumRef, solver.model().eval(z3.fpToIEEEBV(probe), model_completion=True))
    return bits.as_long()


def _fp_total_key(bits: int, width: int = 32) -> int:
    sign_mask = 1 << (width - 1)
    full_mask = (1 << width) - 1
    if bits & sign_mask:
        return (~bits) & full_mask
    return bits ^ sign_mask


def _is_nan_bits(bits: int, ebits: int = 8, sbits: int = 24) -> bool:
    frac_bits = sbits - 1
    exponent = (bits >> frac_bits) & ((1 << ebits) - 1)
    significand = bits & ((1 << frac_bits) - 1)
    return exponent == (1 << ebits) - 1 and significand != 0


class TestFPOMTParser(unittest.TestCase):
    """Parser tests for FP optimization problems."""

    def test_parse_single_fp_objective(self):
        parser = FPOMTParser()
        parser.parse_with_z3(
            """
            (set-logic QF_FP)
            (declare-fun x () (_ FloatingPoint 8 24))
            (define-fun zero () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.0))
            (define-fun one () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 1.0))
            (assert (or (fp.eq x zero) (fp.eq x one)))
            (maximize x)
            (check-sat)
            """
        )

        self.assertEqual(len(parser.objectives), 1)
        self.assertEqual(parser.original_directions, ["max"])
        self.assertIsNotNone(parser.objective)
        objective = cast(z3.ExprRef, parser.objective)
        self.assertEqual(objective.sort_kind(), z3.Z3_FLOATING_POINT_SORT)

    def test_parse_multi_objective_fp_instance(self):
        parser = FPOMTParser()
        parser.parse_with_z3(
            """
            (set-logic QF_FP)
            (declare-fun x () (_ FloatingPoint 8 24))
            (declare-fun y () (_ FloatingPoint 8 24))
            (define-fun zero () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.0))
            (assert (and (fp.eq x zero) (fp.eq y zero)))
            (maximize x)
            (minimize y)
            (check-sat)
            """
        )

        self.assertEqual(len(parser.objectives), 2)
        self.assertEqual(parser.original_directions, ["max", "min"])


class TestFPOMTIterativeSearch(unittest.TestCase):
    """Optimization tests for floating-point objectives."""

    def setUp(self):
        self.sort = z3.FPSort(8, 24)
        self.rne = z3.RNE()
        self.neg_two = z3.FPVal(-2.0, self.sort)
        self.neg_zero = z3.FPVal(-0.0, self.sort)
        self.zero = z3.FPVal(0.0, self.sort)
        self.one = z3.FPVal(1.0, self.sort)
        self.two = z3.FPVal(2.0, self.sort)
        self.pos_nan = z3.fpBVToFP(z3.BitVecVal(0x7FC00001, 32), self.sort)
        self.neg_nan = z3.fpBVToFP(z3.BitVecVal(0xFFC00001, 32), self.sort)

    def test_binary_search_maximize_fp_variable(self):
        x = z3.FP("x", self.sort)
        fml = cast(
            z3.ExprRef,
            z3.Or(
            z3.fpEQ(x, self.neg_two),
            z3.fpEQ(x, self.neg_zero),
            z3.fpEQ(x, self.one),
            ),
        )

        result = fp_opt_with_binary_search(fml, x, minimize=False)

        self.assertIsNotNone(result)
        self.assertEqual(_fp_bits(cast(z3.ExprRef, result)), _fp_bits(self.one))

    def test_linear_search_minimize_fp_variable(self):
        x = z3.FP("x", self.sort)
        fml = cast(
            z3.ExprRef,
            z3.Or(
            z3.fpEQ(x, self.neg_two),
            z3.fpEQ(x, self.neg_zero),
            z3.fpEQ(x, self.one),
            ),
        )

        result = fp_opt_with_linear_search(fml, x, minimize=True)

        self.assertIsNotNone(result)
        self.assertEqual(_fp_bits(cast(z3.ExprRef, result)), _fp_bits(self.neg_two))

    def test_binary_search_handles_term_objective(self):
        x = z3.FP("x", self.sort)
        y = z3.FP("y", self.sort)
        fml = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(z3.fpEQ(x, self.zero), z3.fpEQ(x, self.one)),
                z3.fpEQ(y, self.one),
            ),
        )
        obj = z3.fpAdd(self.rne, x, y)

        result = fp_opt_with_binary_search(fml, obj, minimize=False)

        self.assertIsNotNone(result)
        self.assertEqual(_fp_bits(cast(z3.ExprRef, result)), _fp_bits(self.two))

    def test_binary_search_excludes_nan_models(self):
        x = z3.FP("x", self.sort)
        xb = z3.BitVec("xb", 32)
        fml = cast(
            z3.ExprRef,
            z3.And(
                x == z3.fpBVToFP(xb, self.sort),
                z3.Or(
                    xb == z3.BitVecVal(_fp_bits(self.one), 32),
                    xb == z3.BitVecVal(_fp_bits(self.pos_nan), 32),
                ),
            ),
        )

        result = fp_opt_with_binary_search(fml, x, minimize=False)

        self.assertIsNotNone(result)
        result_bits = _fp_bits(cast(z3.ExprRef, result))
        self.assertTrue(_is_nan_bits(result_bits))
        self.assertGreater(_fp_total_key(result_bits), _fp_total_key(_fp_bits(self.one)))

    def test_qsmt_respects_total_order_for_signed_zero(self):
        x = z3.FP("x", self.sort)
        fml = cast(
            z3.ExprRef,
            z3.Or(z3.fpToIEEEBV(x) == z3.fpToIEEEBV(self.neg_zero), z3.fpToIEEEBV(x) == z3.fpToIEEEBV(self.zero)),
        )

        min_result = fp_opt_with_qsmt(fml, x, minimize=True)
        max_result = fp_opt_with_qsmt(fml, x, minimize=False)

        self.assertIsNotNone(min_result)
        self.assertIsNotNone(max_result)
        self.assertEqual(_fp_bits(cast(z3.ExprRef, min_result)), _fp_bits(self.neg_zero))
        self.assertEqual(_fp_bits(cast(z3.ExprRef, max_result)), _fp_bits(self.zero))

    def test_qsmt_orders_nan_and_infinity_faithfully(self):
        x = z3.FP("x", self.sort)
        xb = z3.BitVec("xb_qsmt", 32)
        fml = cast(
            z3.ExprRef,
            z3.And(
                x == z3.fpBVToFP(xb, self.sort),
                z3.Or(
                    xb == z3.BitVecVal(_fp_bits(self.neg_nan), 32),
                    xb == z3.BitVecVal(_fp_bits(z3.fpMinusInfinity(self.sort)), 32),
                    xb == z3.BitVecVal(_fp_bits(z3.fpPlusInfinity(self.sort)), 32),
                    xb == z3.BitVecVal(_fp_bits(self.pos_nan), 32),
                ),
            ),
        )

        min_result = fp_opt_with_qsmt(fml, x, minimize=True)
        max_result = fp_opt_with_qsmt(fml, x, minimize=False)

        self.assertIsNotNone(min_result)
        self.assertIsNotNone(max_result)
        min_bits = _fp_bits(cast(z3.ExprRef, min_result))
        max_bits = _fp_bits(cast(z3.ExprRef, max_result))
        self.assertTrue(_is_nan_bits(min_bits))
        self.assertTrue(_is_nan_bits(max_bits))
        self.assertLess(
            _fp_total_key(min_bits),
            _fp_total_key(_fp_bits(z3.fpMinusInfinity(self.sort))),
        )
        self.assertGreater(
            _fp_total_key(max_bits),
            _fp_total_key(_fp_bits(z3.fpPlusInfinity(self.sort))),
        )

    def test_total_key_matches_total_order_samples(self):
        samples = [
            self.neg_nan,
            z3.fpMinusInfinity(self.sort),
            self.neg_two,
            self.neg_zero,
            self.zero,
            self.one,
            z3.fpPlusInfinity(self.sort),
            self.pos_nan,
        ]
        width = self.sort.ebits() + self.sort.sbits()
        keys = []
        for sample in samples:
            keys.append(_fp_total_key(_fp_bits(sample), width))
        self.assertEqual(keys, sorted(keys))

    def test_solve_opt_file_uses_fp_backend(self):
        smt2 = """
        (set-logic QF_FP)
        (declare-fun x () (_ FloatingPoint 8 24))
        (define-fun m2 () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE (- 2.0)))
        (define-fun p1 () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 1.0))
        (assert (or (fp.eq x m2) (fp.eq x p1)))
        (minimize x)
        (check-sat)
        """

        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(smt2)
            filename = handle.name

        try:
            result = solve_opt_file(filename, engine="iter", solver_name="z3-bs")
        finally:
            os.unlink(filename)

        self.assertIsNotNone(result)
        self.assertIn("-1*(2**1)", cast(str, result))

    def test_solve_opt_file_supports_qsmt_fp(self):
        smt2 = """
        (set-logic QF_FP)
        (declare-fun x () (_ FloatingPoint 8 24))
        (assert (or (= (fp.to_ieee_bv x) (fp.to_ieee_bv (_ NaN 8 24)))
                    (= (fp.to_ieee_bv x) (fp.to_ieee_bv (_ -zero 8 24)))))
        (maximize x)
        (check-sat)
        """

        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(smt2)
            filename = handle.name

        try:
            result = solve_opt_file(filename, engine="qsmt", solver_name="z3")
        finally:
            os.unlink(filename)

        self.assertIsNotNone(result)
        self.assertIn("NaN", cast(str, result))

    def test_solve_opt_file_supports_lex_multi_objective_fp(self):
        smt2 = """
        (set-logic QF_FP)
        (declare-fun x () (_ FloatingPoint 8 24))
        (declare-fun y () (_ FloatingPoint 8 24))
        (assert (or (= (fp.to_ieee_bv x) (fp.to_ieee_bv (_ -zero 8 24)))
                    (= (fp.to_ieee_bv x) (fp.to_ieee_bv (_ +zero 8 24)))))
        (assert (or (= (fp.to_ieee_bv y) (fp.to_ieee_bv ((_ to_fp 8 24) RNE 1.0)))
                    (= (fp.to_ieee_bv y) (fp.to_ieee_bv ((_ to_fp 8 24) RNE 2.0)))))
        (minimize x)
        (maximize y)
        (check-sat)
        """

        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(smt2)
            filename = handle.name

        try:
            result = solve_opt_file(
                filename,
                engine="qsmt",
                solver_name="z3",
                opt_priority="lex",
            )
        finally:
            os.unlink(filename)

        self.assertIsNotNone(result)
        result_text = cast(str, result)
        self.assertIn("-0.0", result_text)
        self.assertIn("2", result_text)

    def test_fp_optimize_pareto_enumerates_frontier(self):
        x = z3.FP("x_pareto", self.sort)
        y = z3.FP("y_pareto", self.sort)
        xb = z3.BitVec("xb_pareto", 32)
        yb = z3.BitVec("yb_pareto", 32)
        zero_bits = z3.BitVecVal(_fp_bits(self.zero), 32)
        one_bits = z3.BitVecVal(_fp_bits(self.one), 32)
        two_bits = z3.BitVecVal(_fp_bits(self.two), 32)
        fml = cast(
            z3.ExprRef,
            z3.And(
                x == z3.fpBVToFP(xb, self.sort),
                y == z3.fpBVToFP(yb, self.sort),
                z3.Or(
                    z3.And(xb == zero_bits, yb == two_bits),
                    z3.And(xb == one_bits, yb == one_bits),
                    z3.And(xb == two_bits, yb == zero_bits),
                ),
            ),
        )

        frontier = fp_optimize_pareto(
            fml,
            [x, y],
            ["max", "max"],
            engine="iter",
            solver_name="z3-bs",
        )

        points = {tuple(_fp_bits(value) for value in point) for point in frontier}
        expected = {
            (_fp_bits(self.zero), _fp_bits(self.two)),
            (_fp_bits(self.one), _fp_bits(self.one)),
            (_fp_bits(self.two), _fp_bits(self.zero)),
        }
        self.assertEqual(points, expected)

    def test_solve_opt_file_supports_pareto_multi_objective_fp(self):
        smt2 = """
        (set-logic QF_FP)
        (declare-fun x () (_ FloatingPoint 8 24))
        (declare-fun y () (_ FloatingPoint 8 24))
        (define-fun z0 () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 0.0))
        (define-fun z1 () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 1.0))
        (define-fun z2 () (_ FloatingPoint 8 24) ((_ to_fp 8 24) RNE 2.0))
        (assert (or (and (fp.eq x z0) (fp.eq y z2))
                    (and (fp.eq x z1) (fp.eq y z1))
                    (and (fp.eq x z2) (fp.eq y z0))))
        (maximize x)
        (maximize y)
        (check-sat)
        """

        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(smt2)
            filename = handle.name

        try:
            result = solve_opt_file(
                filename,
                engine="iter",
                solver_name="z3-bs",
                opt_priority="par",
            )
        finally:
            os.unlink(filename)

        self.assertIsNotNone(result)
        result_text = cast(str, result)
        self.assertIn("0.0", result_text)
        self.assertIn("1", result_text)
        self.assertIn("1*(2**1)", result_text)

    def test_solve_opt_file_rejects_unsupported_fp_engines(self):
        smt2 = """
        (set-logic QF_FP)
        (declare-fun x () (_ FloatingPoint 8 24))
        (assert (= (fp.to_ieee_bv x) (fp.to_ieee_bv ((_ to_fp 8 24) RNE 0.0))))
        (maximize x)
        (check-sat)
        """

        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(smt2)
            filename = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "does not support MaxSAT"):
                solve_opt_file(filename, engine="maxsat", solver_name="FM")
            with self.assertRaisesRegex(ValueError, "does not support floating-point"):
                solve_opt_file(filename, engine="z3py", solver_name="z3py")
        finally:
            os.unlink(filename)


if __name__ == "__main__":
    unittest.main()
