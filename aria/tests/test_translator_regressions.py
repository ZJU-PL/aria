import contextlib
import importlib
import io
import os
import subprocess
import sys
import tempfile

import z3

from aria.tests import TestCase, main
from aria.translator import cnf2smt, dimacs2smt, qbf2smt


class TestTranslatorRegressions(TestCase):
    def test_cnf2lp_cli_without_arguments_prints_usage(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.translator.cnf2lp"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage", result.stdout.lower())

    def test_dimacs_empty_clause_preserved_as_unsat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cnf_path = os.path.join(temp_dir, "in.cnf")
            smt_path = os.path.join(temp_dir, "out.smt2")

            with open(cnf_path, "w", encoding="utf-8") as fd:
                fd.write("p cnf 1 1\n0\n")

            dimacs2smt.convert_dimacs_to_smt2(cnf_path, smt_path)

            with open(smt_path, "r", encoding="utf-8") as fd:
                smt = fd.read()

            self.assertIn("(assert false)", smt)

    def test_qbf_parser_accepts_blank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            qbf_path = os.path.join(temp_dir, "in.qdimacs")
            with open(qbf_path, "w", encoding="utf-8") as fd:
                fd.write("p cnf 1 1\na 1 0\n\n1 0\n")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                ret = qbf2smt.parse(qbf_path)

            self.assertEqual(ret, 0)
            self.assertIn("(check-sat)", stdout.getvalue())

    def test_qdimacs_file_translation_writes_parseable_smt2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            qbf_path = os.path.join(temp_dir, "in.qdimacs")
            smt_path = os.path.join(temp_dir, "out.smt2")
            with open(qbf_path, "w", encoding="utf-8") as fd:
                fd.write("p cnf 1 1\ne 1 0\n1 0\n")

            qbf2smt.convert_qdimacs_to_smt2(qbf_path, smt_path)

            solver = z3.Solver()
            solver.add(z3.parse_smt2_file(smt_path))
            self.assertEqual(solver.check(), z3.sat)

    def test_qdimacs_empty_clause_translates_to_unsat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            qbf_path = os.path.join(temp_dir, "unsat.qdimacs")
            smt_path = os.path.join(temp_dir, "unsat.smt2")
            with open(qbf_path, "w", encoding="utf-8") as fd:
                fd.write("p cnf 1 1\ne 1 0\n0\n")

            qbf2smt.convert_qdimacs_to_smt2(qbf_path, smt_path)

            solver = z3.Solver()
            solver.add(z3.parse_smt2_file(smt_path))
            self.assertEqual(solver.check(), z3.unsat)

    def test_cnf2smt_shared_parser_handles_blank_lines(self):
        dimacs_str = "p cnf 1 1\n\n1 0\n"
        num_vars, num_clauses, clauses = cnf2smt.parse_dimacs_string(dimacs_str)

        self.assertEqual(num_vars, 1)
        self.assertEqual(num_clauses, 1)
        self.assertEqual(clauses, [[1]])

    def test_fzn2omt_modules_import_from_package(self):
        module_names = [
            "aria.translator.fzn2omt.fzn2z3",
            "aria.translator.fzn2omt.fzn2cvc4",
            "aria.translator.fzn2omt.fzn2optimathsat",
            "aria.translator.fzn2omt.smt2model2fzn",
        ]

        for module_name in module_names:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    main()
