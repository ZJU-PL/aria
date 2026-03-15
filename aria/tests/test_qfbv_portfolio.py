import json
import multiprocessing
import os
import shutil
import sys
import tempfile

from aria.tests import TestCase, main
from aria.smt.portfolio.qfbv_portfolio import (
    FormulaParser,
    SolverConfig,
    SolverResult,
    _preprocess_and_solve_sat_worker,
    _wait_for_first_result,
)


def _write_smt2_file(contents: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False, encoding="utf-8"
    )
    handle.write(contents)
    handle.close()
    return handle.name


class TestQFBVPortfolio(TestCase):
    def test_wait_for_first_result_times_out(self):
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        try:
            result = _wait_for_first_result(result_queue, 0.01)
            self.assertIsNone(result)
        finally:
            result_queue.close()
            result_queue.join_thread()

    def test_solve_detailed_sat_under_spawn(self):
        formula_file = _write_smt2_file(
            "\n".join(
                [
                    "(set-logic QF_BV)",
                    "(declare-fun x () (_ BitVec 8))",
                    "(assert (= x #x0a))",
                    "(check-sat)",
                ]
            )
        )
        self.addCleanup(os.unlink, formula_file)

        result = FormulaParser.solve_detailed(
            formula_file, config=SolverConfig(start_method="spawn")
        )

        self.assertEqual(result.result, SolverResult.SAT)
        self.assertIsNotNone(result.winner)

    def test_solve_detailed_unsat_under_spawn(self):
        formula_file = _write_smt2_file(
            "\n".join(
                [
                    "(set-logic QF_BV)",
                    "(declare-fun x () (_ BitVec 8))",
                    "(assert (= x #x0a))",
                    "(assert (= x #x0b))",
                    "(check-sat)",
                ]
            )
        )
        self.addCleanup(os.unlink, formula_file)

        result = FormulaParser.solve_detailed(
            formula_file, config=SolverConfig(start_method="spawn")
        )

        self.assertEqual(result.result, SolverResult.UNSAT)
        self.assertIsNotNone(result.winner)

    def test_solve_detailed_invalid_smt2_returns_unknown(self):
        formula_file = _write_smt2_file("(set-logic QF_BV)\n(assert invalid)\n")
        self.addCleanup(os.unlink, formula_file)

        result = FormulaParser.solve_detailed(formula_file)

        self.assertEqual(result.result, SolverResult.UNKNOWN)
        self.assertIsNotNone(result.error)

    def test_preprocess_worker_reports_unknown_on_invalid_preamble(self):
        formula_file = _write_smt2_file(
            "\n".join(
                [
                    "(set-logic QF_BV)",
                    "(declare-fun x () (_ BitVec 8))",
                    "(assert (= x #x0a))",
                    "(check-sat)",
                ]
            )
        )
        self.addCleanup(os.unlink, formula_file)

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        try:
            _preprocess_and_solve_sat_worker(
                formula_file,
                999,
                [],
                0.01,
                "spawn",
                result_queue,
            )
            result = result_queue.get(timeout=1.0)
        finally:
            result_queue.close()
            result_queue.join_thread()

        self.assertEqual(result["result"], SolverResult.UNKNOWN.value)
        self.assertIn("Unknown Z3 preamble id", result["error"])

    def test_main_writes_solver_output_and_artifacts(self):
        formula_file = _write_smt2_file(
            "\n".join(
                [
                    "(set-logic QF_BV)",
                    "(declare-fun x () (_ BitVec 8))",
                    "(assert (= x #x0a))",
                    "(check-sat)",
                ]
            )
        )
        self.addCleanup(os.unlink, formula_file)

        request_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, request_directory)

        with open(
            os.path.join(request_directory, "input.json"), "w", encoding="utf-8"
        ) as input_file:
            json.dump({"formula_file": formula_file}, input_file)

        from aria.smt.portfolio import qfbv_portfolio

        previous_argv = sys.argv
        sys.argv = ["qfbv_portfolio.py", request_directory]
        try:
            qfbv_portfolio.main()
        finally:
            sys.argv = previous_argv

        with open(
            os.path.join(request_directory, "solver_out.json"), encoding="utf-8"
        ) as output_file:
            solver_output = json.load(output_file)

        self.assertEqual(solver_output["return_code"], SolverResult.SAT.return_code)
        self.assertTrue(os.path.exists(solver_output["artifacts"]["stdout_path"]))
        self.assertTrue(os.path.exists(solver_output["artifacts"]["stderr_path"]))


if __name__ == "__main__":
    main()
