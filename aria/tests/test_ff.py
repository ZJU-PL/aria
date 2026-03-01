import tempfile

import z3

from aria.smt.ff.ff_ast import FieldConst, FieldEq, FieldVar, ParsedFormula
from aria.smt.ff.ff_bv_solver import FFBVSolver
from aria.smt.ff.ff_bv_solver2 import FFBVBridgeSolver
from aria.smt.ff.ff_int_solver import FFIntSolver
from aria.smt.ff.ff_parser import parse_ff_file, parse_ff_file_strict
from aria.smt.ff.ff_solver import FFAutoSolver
from aria.tests import TestCase, main


def _parse_text(smt_text: str) -> ParsedFormula:
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
        handle.write(smt_text)
        path = handle.name
    return parse_ff_file(path)


class TestFiniteFieldSMT(TestCase):
    def test_parser_supports_macros_bitsum_and_multi_field(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F3 () (_ FiniteField 3))
            (define-sort F5 () (_ FiniteField 5))
            (declare-const a F3)
            (declare-const b F5)
            (define-fun is_bit ((f F3)) Bool
              (or (= f (as ff0 F3)) (= f (as ff1 F3))))
            (assert
              (or
                (is_bit a)
                (= (ff.bitsum (as ff-1 F5) b) b)))
            (check-sat)
            """
        )
        self.assertEqual(formula.field_sizes, [3, 5])
        self.assertIsNone(formula.field_size)

        solver = FFIntSolver()
        self.assertEqual(solver.check(formula), z3.sat)

    def test_strict_parser_rejects_mixed_fields(self):
        with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as handle:
            handle.write(
                """
                (set-logic QF_FF)
                (define-sort F3 () (_ FiniteField 3))
                (define-sort F5 () (_ FiniteField 5))
                (declare-const a F3)
                (declare-const b F5)
                (assert (or (= a (as ff0 F3)) (= b (as ff0 F5))))
                (check-sat)
                """
            )
            path = handle.name
        with self.assertRaises(ValueError):
            parse_ff_file_strict(path)

    def test_solver_instances_are_reusable(self):
        formula_one = ParsedFormula(
            5,
            {"x": "ff:5"},
            [FieldEq(FieldVar("x"), FieldConst(1, 5))],
            field_sizes=[5],
        )
        formula_two = ParsedFormula(
            5,
            {"x": "ff:5"},
            [FieldEq(FieldVar("x"), FieldConst(2, 5))],
            field_sizes=[5],
        )

        for solver_class in (FFIntSolver, FFBVSolver, FFBVBridgeSolver):
            solver = solver_class()
            self.assertEqual(solver.check(formula_one), z3.sat)
            self.assertEqual(solver.check(formula_two), z3.sat)

    def test_booleanity_rewrite_makes_constraint_explicit(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const x F)
            (assert (= (ff.mul x (ff.add x (as ff-1 F))) (as ff0 F)))
            (assert (not (or (= x (as ff0 F)) (= x (as ff1 F)))))
            (check-sat)
            """
        )
        self.assertEqual(FFAutoSolver().check(formula), z3.unsat)

    def test_bitsum_negative_constants(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F2 () (_ FiniteField 2))
            (assert (= (as ff-9 F2) (ff.bitsum (as ff-9 F2) (as ff-10 F2))))
            (check-sat)
            """
        )
        self.assertEqual(FFAutoSolver().check(formula), z3.sat)

    def test_auto_solver_backend_selection(self):
        small = _parse_text(
            """
            (set-logic QF_FF)
            (declare-const x (_ FiniteField 17))
            (assert (= x x))
            (check-sat)
            """
        )
        medium = _parse_text(
            """
            (set-logic QF_FF)
            (declare-const x (_ FiniteField 2305843009213693951))
            (assert (= x x))
            (check-sat)
            """
        )
        large = _parse_text(
            """
            (set-logic QF_FF)
            (declare-const x (_ FiniteField 52435875175126190479447740508185965837690552500527637822603658699938581184513))
            (assert (= x x))
            (check-sat)
            """
        )

        solver = FFAutoSolver()
        self.assertEqual(solver.check(small), z3.sat)
        self.assertEqual(solver.backend_name, "bv")

        solver = FFAutoSolver()
        self.assertEqual(solver._select_backend(medium), "bv2")

        solver = FFAutoSolver()
        self.assertEqual(solver._select_backend(large), "int")


if __name__ == "__main__":
    main()
