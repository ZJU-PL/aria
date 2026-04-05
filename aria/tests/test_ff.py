import tempfile
import os
import json
import subprocess
import sys

import z3

from aria.smt.ff.core.ff_algebra import FFLocalAlgebraicReasoner
from aria.smt.ff.core.ff_ast import (
    BoolOr,
    FieldAdd,
    FieldConst,
    FieldEq,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldVar,
    ParsedFormula,
)
from aria.smt.ff.core.ff_modkernels import ModKernelSelector, ModReducer
from aria.smt.ff.core.ff_poly import (
    partition_polynomial_assertions,
    polynomial_from_equality,
    polynomial_from_expr,
)
from aria.smt.ff.frontend.ff_parser import parse_ff_file, parse_ff_file_strict
from aria.smt.ff.frontend.ff_preprocess import (
    preprocess_formula,
    preprocess_formula_with_metadata,
)
from aria.smt.ff.solvers.ff_bv_solver import FFBVSolver
from aria.smt.ff.solvers.ff_bv_solver2 import FFBVBridgeSolver
from aria.smt.ff.solvers.ff_int_solver import FFIntSolver
from aria.smt.ff.solvers.ff_perf_solver import FFPerfSolver
from aria.smt.ff.solvers.ff_solver import FFAutoSolver
from aria.tests import TestCase, main

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


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
        self.assertEqual(solver._select_backend(large), "perf")

        solver = FFAutoSolver(enable_perf_backend=False)
        self.assertEqual(solver._select_backend(large), "int")

    def test_perf_solver_basic_sat_and_stats(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const x F)
            (assert (= (ff.mul x x) (as ff1 F)))
            (check-sat)
            """
        )
        solver = FFPerfSolver()
        self.assertEqual(solver.check(formula), z3.sat)
        stats = solver.stats()
        self.assertGreater(stats.get("reductions_total", 0), 0)

    def test_perf_solver_recovery_path(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const x F)
            (assert (= (ff.add x (as ff1 F)) (as ff2 F)))
            (check-sat)
            """
        )
        solver = FFPerfSolver(schedule="lazy", recovery=True)
        self.assertEqual(solver.check(formula), z3.sat)
        self.assertGreaterEqual(solver.stats().get("fallback_attempts", 0), 0)

    def test_perf_solver_cegar_refines_spurious_cut_model(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 5))
            (declare-const x F)
            (assert (= (ff.mul x x) x))
            (assert (not (or (= x (as ff0 F)) (= x (as ff1 F)))))
            (check-sat)
            """
        )
        solver = FFPerfSolver(schedule="lazy", recovery=True, max_refinement_rounds=4)
        self.assertEqual(solver.check(formula), z3.unsat)
        stats = solver.stats()
        self.assertGreater(stats.get("validation_failures", 0), 0)
        self.assertTrue(
            stats.get("cuts_defined", 0) > 0 or stats.get("lemmas_learned", 0) > 0
        )

    def test_perf_solver_honors_env(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const x F)
            (assert (= x (as ff3 F)))
            (check-sat)
            """
        )
        old_schedule = os.environ.get("ARIA_FF_SCHEDULE")
        old_kernel = os.environ.get("ARIA_FF_KERNEL_MODE")
        try:
            os.environ["ARIA_FF_SCHEDULE"] = "eager"
            os.environ["ARIA_FF_KERNEL_MODE"] = "generic"
            solver = FFPerfSolver()
            self.assertEqual(solver.check(formula), z3.sat)
            self.assertEqual(solver.initial_schedule, "eager")
            self.assertEqual(solver.kernel_mode, "generic")
        finally:
            if old_schedule is None:
                os.environ.pop("ARIA_FF_SCHEDULE", None)
            else:
                os.environ["ARIA_FF_SCHEDULE"] = old_schedule
            if old_kernel is None:
                os.environ.pop("ARIA_FF_KERNEL_MODE", None)
            else:
                os.environ["ARIA_FF_KERNEL_MODE"] = old_kernel

    def test_mod_kernel_reduction_equivalence(self):
        moduli = [17, 31, 127]
        values = [0, 1, 7, 15, 63, 127, 255, 1024, 99991]
        for modulus in moduli:
            spec = ModKernelSelector(kernel_mode="structured").classify(modulus)
            reducer = ModReducer({modulus: spec})
            for value in values:
                reduced = reducer.reduce(z3.IntVal(value), modulus)
                simplify_val = z3.simplify(reduced).as_long()
                self.assertEqual(simplify_val, value % modulus)

    def test_polynomial_ir_normalizes_and_round_trips(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 7))
            (declare-const x F)
            (declare-const y F)
            (assert (= (ff.add (ff.mul x x) (ff.mul (as ff2 F) x y) (as ff8 F))
                       (ff.add (ff.mul y (as ff0 F)) (as ff1 F))))
            (check-sat)
            """
        )
        poly = polynomial_from_equality(formula.assertions[0], formula.variables)
        self.assertIsNotNone(poly)
        self.assertEqual(poly.modulus, 7)
        self.assertEqual(poly.constant_term(), 0)
        rebuilt = poly.to_expr()
        rebuilt_poly = polynomial_from_expr(rebuilt, formula.variables)
        self.assertEqual(rebuilt_poly.terms, poly.terms)

    def test_preprocess_metadata_reports_structured_partitions(self):
        formula = ParsedFormula(
            17,
            {"x": "ff:17", "z": "ff:17"},
            [
                FieldEq(FieldPow(FieldVar("x"), 3), FieldConst(0, 17)),
                FieldEq(
                    FieldAdd(FieldVar("z"), FieldConst(2, 17)),
                    FieldConst(0, 17),
                ),
            ],
            field_sizes=[17],
        )
        normalized, metadata = preprocess_formula_with_metadata(formula)
        partitions = partition_polynomial_assertions(
            normalized.assertions, normalized.variables
        )
        self.assertGreaterEqual(metadata["structured_rewrites"], 1)
        self.assertEqual(metadata["polynomial_partitions"], 2)
        self.assertEqual(len(partitions), 2)

    def test_preprocess_rewrites_linear_patterns_and_deduplicates(self):
        formula = ParsedFormula(
            17,
            {"x": "ff:17", "y": "ff:17", "b": "ff:17"},
            [
                FieldEq(FieldMul(FieldConst(3, 17), FieldVar("x")), FieldConst(0, 17)),
                FieldEq(FieldAdd(FieldVar("y"), FieldConst(2, 17)), FieldConst(0, 17)),
                FieldEq(FieldAdd(FieldVar("x"), FieldNeg(FieldVar("y"))), FieldConst(0, 17)),
                FieldEq(FieldPow(FieldVar("b"), 2), FieldVar("b")),
                FieldEq(FieldAdd(FieldVar("y"), FieldConst(2, 17)), FieldConst(0, 17)),
            ],
            field_sizes=[17],
        )
        normalized, metadata = preprocess_formula_with_metadata(formula)
        self.assertGreater(metadata["duplicate_assertions_removed"], 0)
        self.assertGreaterEqual(metadata["affine_assertions"], 2)
        self.assertGreaterEqual(len(metadata["affine_assertion_indices"]), 2)
        self.assertEqual(FFAutoSolver().check(normalized), z3.unsat)

    def test_local_reasoner_derives_linear_root(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const z F)
            (assert (= (ff.add (ff.mul (as ff3 F) z) (as ff4 F)) (as ff0 F)))
            (check-sat)
            """
        )
        lemmas = FFLocalAlgebraicReasoner().derive_lemmas(
            formula.assertions[0], formula.variables
        )
        kinds = {lemma.kind for lemma in lemmas}
        self.assertIn("linear-root", kinds)
        root_lemmas = [lemma for lemma in lemmas if lemma.kind == "linear-root"]
        self.assertEqual(len(root_lemmas), 1)
        self.assertEqual(
            polynomial_from_equality(root_lemmas[0].expr, formula.variables).terms,
            {(("z", 1),): 1, (): 7},
        )

    def test_local_reasoner_derives_affine_partition_roots(self):
        formula = ParsedFormula(
            17,
            {"x": "ff:17", "y": "ff:17"},
            [
                FieldEq(
                    FieldAdd(FieldVar("x"), FieldVar("y")),
                    FieldConst(3, 17),
                ),
                FieldEq(FieldVar("x"), FieldConst(1, 17)),
            ],
            field_sizes=[17],
        )
        lemmas = FFLocalAlgebraicReasoner().derive_partition_lemmas(
            formula.assertions, formula.variables
        )
        roots = {
            tuple(
                sorted(
                    polynomial_from_equality(lemma.expr, formula.variables).terms.items()
                )
            )
            for lemma in lemmas
            if lemma.kind == "affine-root"
        }
        self.assertIn(
            tuple(sorted({(("x", 1),): 1, (): 16}.items())),
            roots,
        )
        self.assertIn(
            tuple(sorted({(("y", 1),): 1, (): 15}.items())),
            roots,
        )

    def test_local_reasoner_derives_nonlinear_partition_rootset(self):
        formula = ParsedFormula(
            5,
            {"x": "ff:5"},
            [
                FieldEq(
                    FieldMul(FieldVar("x"), FieldVar("x")),
                    FieldConst(1, 5),
                )
            ],
            field_sizes=[5],
        )
        lemmas = FFLocalAlgebraicReasoner().derive_partition_lemmas(
            formula.assertions, formula.variables
        )
        rootset_terms = [
            lemma
            for lemma in lemmas
            if lemma.kind == "partition-rootset"
        ]
        self.assertEqual(len(rootset_terms), 1)
        expr = rootset_terms[0].expr
        self.assertIsInstance(expr, BoolOr)
        self.assertEqual(
            {
                tuple(sorted(polynomial_from_equality(arg, formula.variables).terms.items()))
                for arg in expr.args
            },
            {
                tuple(sorted({(("x", 1),): 1, (): 4}.items())),
                tuple(sorted({(("x", 1),): 1, (): 1}.items())),
            },
        )

    def test_local_reasoner_derives_nonlinear_partition_contradiction(self):
        formula = ParsedFormula(
            5,
            {"x": "ff:5"},
            [
                FieldEq(
                    FieldMul(FieldVar("x"), FieldVar("x")),
                    FieldConst(2, 5),
                )
            ],
            field_sizes=[5],
        )
        lemmas = FFLocalAlgebraicReasoner().derive_partition_lemmas(
            formula.assertions, formula.variables
        )
        self.assertIn("partition-contradiction", {lemma.kind for lemma in lemmas})

    def test_local_reasoner_partition_cache_hits(self):
        formula = ParsedFormula(
            5,
            {"x": "ff:5"},
            [
                FieldEq(
                    FieldMul(FieldVar("x"), FieldVar("x")),
                    FieldConst(1, 5),
                )
            ],
            field_sizes=[5],
        )
        reasoner = FFLocalAlgebraicReasoner()
        reasoner.derive_partition_lemmas(formula.assertions, formula.variables)
        reasoner.derive_partition_lemmas(formula.assertions, formula.variables)
        stats = reasoner.stats()
        self.assertGreater(stats.get("partition_cache_hits", 0), 0)
        self.assertGreater(stats.get("partition_cache_misses", 0), 0)

    def test_perf_solver_learns_zero_product_explanation_lemma(self):
        formula = _parse_text(
            """
            (set-logic QF_FF)
            (define-sort F () (_ FiniteField 17))
            (declare-const x F)
            (declare-const y F)
            (assert (= (ff.mul (ff.add x (as ff1 F)) (ff.add y (as ff2 F)))
                       (as ff0 F)))
            (assert (not (= x (as ff16 F))))
            (assert (not (= y (as ff15 F))))
            (check-sat)
            """
        )
        solver = FFPerfSolver(
            schedule="lazy",
            recovery=True,
            max_refinement_rounds=4,
            cut_seed_budget=1,
            lemma_refine_budget=1,
        )
        self.assertEqual(solver.check(formula), z3.unsat)
        stats = solver.stats()
        self.assertGreater(stats.get("lemmas_learned", 0), 0)
        self.assertGreater(stats.get("lemma_zero_product", 0), 0)

    def test_perf_solver_records_affine_partition_lemmas(self):
        formula = ParsedFormula(
            17,
            {"x": "ff:17", "y": "ff:17"},
            [
                FieldEq(
                    FieldAdd(FieldVar("x"), FieldVar("y")),
                    FieldConst(3, 17),
                ),
                FieldEq(FieldVar("x"), FieldConst(1, 17)),
                FieldEq(
                    FieldMul(FieldVar("y"), FieldVar("y")),
                    FieldConst(4, 17),
                ),
            ],
            field_sizes=[17],
        )
        normalized = preprocess_formula(formula)
        solver = FFPerfSolver(lemma_refine_budget=4)
        solver._reset_stats()
        solver.var_sorts = dict(normalized.variables)
        solver._learned_lemmas = []
        solver._learned_lemma_keys = set()
        solver._partition_order = solver._build_partition_order(normalized)
        self.assertTrue(solver._learn_explanation_lemmas(normalized.assertions[:2]))
        stats = solver.stats()
        self.assertGreater(stats.get("lemma_affine_root", 0), 0)
        self.assertGreater(stats.get("lemmas_learned", 0), 0)

    def test_perf_solver_tracks_partition_refinement_stats(self):
        formula = ParsedFormula(
            17,
            {"x": "ff:17", "y": "ff:17", "z": "ff:17"},
            [
                FieldEq(
                    FieldMul(
                        FieldAdd(FieldVar("x"), FieldConst(1, 17)),
                        FieldAdd(FieldVar("y"), FieldConst(2, 17)),
                    ),
                    FieldConst(0, 17),
                ),
                FieldEq(FieldVar("z"), FieldConst(3, 17)),
                FieldEq(FieldVar("x"), FieldConst(0, 17)),
                FieldEq(FieldVar("y"), FieldConst(0, 17)),
            ],
            field_sizes=[17],
        )
        solver = FFPerfSolver(schedule="lazy", recovery=True, max_refinement_rounds=4)
        self.assertEqual(solver.check(formula), z3.unsat)
        stats = solver.stats()
        self.assertGreaterEqual(stats.get("partitions_total", 0), 1)
        self.assertGreater(stats.get("partition_refinements", 0), 0)
        self.assertGreater(stats.get("partitions_failed", 0), 0)
        self.assertGreater(stats.get("partition_exactness_gain", 0), 0)

    def test_perf_solver_exposes_trace_and_reasoner_stats(self):
        formula = ParsedFormula(
            5,
            {"x": "ff:5"},
            [
                FieldEq(
                    FieldMul(FieldVar("x"), FieldVar("x")),
                    FieldConst(2, 5),
                )
            ],
            field_sizes=[5],
        )
        solver = FFPerfSolver(schedule="lazy", recovery=True, max_refinement_rounds=3)
        self.assertEqual(solver.check(formula), z3.unsat)
        stats = solver.stats()
        trace = solver.trace()
        self.assertGreater(stats.get("partition_solver_hits", 0), 0)
        self.assertGreaterEqual(stats.get("selected_partition_size", 0), 1)
        self.assertGreaterEqual(len(trace), 1)
        self.assertIn("round", trace[0])

    def test_benchmark_runner_emits_stats(self):
        with tempfile.TemporaryDirectory() as bench_dir:
            bench_path = os.path.join(bench_dir, "tiny.smt2")
            with open(bench_path, "w", encoding="utf-8") as handle:
                handle.write(
                    """
                    (set-logic QF_FF)
                    (define-sort F () (_ FiniteField 5))
                    (declare-const x F)
                    (assert (= (ff.mul x x) (as ff2 F)))
                    (check-sat)
                    """
                )
            out_path = os.path.join(bench_dir, "bench.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_ff_perf_bench.py",
                    "--bench-dir",
                    bench_dir,
                    "--backends",
                    "perf",
                    "--timeouts",
                    "2",
                    "--repetitions",
                    "1",
                    "--out",
                    out_path,
                ],
                 cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            run_data = payload["runs"]["2"]["perf"][bench_path]
            self.assertIn("avg_stats", run_data)
            self.assertIn("avg_trace_length", run_data)
            self.assertIn("avg_stats", payload["summary"]["2"]["perf"])

    def test_perf_solver_learns_nonlinear_partition_contradiction(self):
        formula = ParsedFormula(
            5,
            {"x": "ff:5"},
            [
                FieldEq(
                    FieldMul(FieldVar("x"), FieldVar("x")),
                    FieldConst(2, 5),
                )
            ],
            field_sizes=[5],
        )
        solver = FFPerfSolver(schedule="lazy", recovery=True, max_refinement_rounds=3)
        self.assertEqual(solver.check(formula), z3.unsat)
        stats = solver.stats()
        self.assertGreater(stats.get("lemma_partition_contradiction", 0), 0)
        self.assertGreater(stats.get("partition_lemma_rounds", 0), 0)


if __name__ == "__main__":
    main()
