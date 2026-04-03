"""Tests for aria.cli.efsmt_cli - Exists-Forall SMT CLI."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import z3

from aria.cli.efsmt_cli import main, _infer_theory, _format_result


# Sample EFSMT problems
EFSMT_BOOL_SAMPLE = """(set-logic QF_UF)
(declare-fun x () Bool)
(assert (forall ((y Bool)) (or x y)))
(check-sat)
"""

EFSMT_BV_SAMPLE = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (forall ((y (_ BitVec 8))) (bvule y x)))
(check-sat)
"""

EFSMT_LIRA_SAMPLE = """(set-logic QF_LIA)
(declare-fun x () Int)
(assert (forall ((y Int)) (=> (>= y 0) (>= x y))))
(check-sat)
"""

EFSMT_TRIVIAL_SAMPLE = """(set-logic QF_UF)
(declare-fun x () Bool)
(assert x)
(check-sat)
"""


def _write_file(tmp_path, content, suffix="smt2"):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


class TestInferTheory:
    """Tests for _infer_theory function."""

    def test_infer_bool_theory(self):
        exists_vars = []
        forall_vars = []
        phi = z3.Bool("x")
        result = _infer_theory(exists_vars, forall_vars, phi)
        assert result == "bool"

    def test_infer_bv_theory(self):
        x = z3.BitVec("x", 8)
        exists_vars = [x]
        forall_vars = []
        phi = z3.BoolVal(True)
        result = _infer_theory(exists_vars, forall_vars, phi)
        assert result == "bv"

    def test_infer_lira_theory_int(self):
        x = z3.Int("x")
        exists_vars = [x]
        forall_vars = []
        phi = z3.BoolVal(True)
        result = _infer_theory(exists_vars, forall_vars, phi)
        assert result == "lira"

    def test_infer_lira_theory_real(self):
        x = z3.Real("x")
        exists_vars = [x]
        forall_vars = []
        phi = z3.BoolVal(True)
        result = _infer_theory(exists_vars, forall_vars, phi)
        assert result == "lira"


class TestFormatResult:
    """Tests for _format_result function."""

    def test_format_z3_sat(self):
        assert _format_result(z3.sat) == "sat"

    def test_format_z3_unsat(self):
        assert _format_result(z3.unsat) == "unsat"

    def test_format_z3_unknown(self):
        assert _format_result(z3.unknown) == "unknown"

    def test_format_str_result(self):
        assert _format_result("sat") == "sat"
        assert _format_result("unsat") == "unsat"
        assert _format_result("unknown") == "unknown"


class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys):
        with patch.object(sys, "argv", ["efsmt", "/nonexistent/file.smt2"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_trivial_bool_sat(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, EFSMT_TRIVIAL_SAMPLE)
        with patch.object(sys, "argv", ["efsmt", str(in_file), "--theory", "bool"]):
            result = main()
        # Should complete without error
        assert result in (0, 1)

    def test_main_with_parser_option(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--parser", "z3"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_with_timeout(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--timeout", "10"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_with_max_loops(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--max-loops", "100"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_with_log_level(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--log-level", "DEBUG"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_auto_theory(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--theory", "auto"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_bv_theory(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BV_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--theory", "bv"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_lira_theory(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_LIRA_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--theory", "lira"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_bv_parallel_exposes_sampling_options(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BV_SAMPLE)
        x = z3.BitVec("x", 8)
        y = z3.BitVec("y", 8)
        phi = z3.BoolVal(True)

        with patch("aria.cli.efsmt_cli._parse_efsmt_file", return_value=([x], [y], phi)):
            with patch("aria.cli.efsmt_cli.ParallelEFBVSolver") as solver_cls:
                solver_cls.return_value.solve_efsmt_bv.return_value = "sat"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "efsmt",
                        str(in_file),
                        "--theory",
                        "bv",
                        "--engine",
                        "efbv-par",
                        "--max-loops",
                        "12",
                        "--efbv-num-samples",
                        "7",
                    ],
                ):
                    result = main()

        assert result == 0
        solver_cls.assert_called_once_with(mode="canary", maxloops=12, num_samples=7)

    def test_main_bv_seq_exposes_pysmt_solver(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BV_SAMPLE)
        x = z3.BitVec("x", 8)
        y = z3.BitVec("y", 8)
        phi = z3.BoolVal(True)

        with patch("aria.cli.efsmt_cli._parse_efsmt_file", return_value=([x], [y], phi)):
            with patch("aria.cli.efsmt_cli.EFBVSequentialSolver") as solver_cls:
                solver_cls.return_value.solve.return_value = "sat"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "efsmt",
                        str(in_file),
                        "--theory",
                        "bv",
                        "--engine",
                        "efbv-seq",
                        "--bv-solver",
                        "cegis",
                        "--bv-pysmt-solver",
                        "cvc5",
                    ],
                ):
                    result = main()

        assert result == 0
        solver_cls.assert_called_once_with("BV", solver="cegis", pysmt_solver="cvc5")

    def test_main_lira_parallel_exposes_engine_options(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_LIRA_SAMPLE)
        x = z3.Int("x")
        y = z3.Int("y")
        phi = z3.BoolVal(True)

        with patch("aria.cli.efsmt_cli._parse_efsmt_file", return_value=([x], [y], phi)):
            with patch("aria.cli.efsmt_cli.ParallelEFLIRASolver") as solver_cls:
                solver_cls.return_value.solve_efsmt_lira.return_value = "sat"
                with patch.object(
                    sys,
                    "argv",
                    [
                        "efsmt",
                        str(in_file),
                        "--theory",
                        "lira",
                        "--engine",
                        "eflira-par",
                        "--forall-solver",
                        "cvc5",
                        "--eflira-forall-mode",
                        "parallel-process-ipc",
                        "--eflira-num-workers",
                        "8",
                        "--eflira-num-samples",
                        "6",
                        "--eflira-sample-strategy",
                        "lexicographic",
                        "--eflira-sample-max-tries",
                        "40",
                        "--eflira-sample-seed-low",
                        "3",
                        "--eflira-sample-seed-high",
                        "99",
                        "--eflira-lex-order",
                        "x,y",
                        "--eflira-jitter-real-delta",
                        "0.25",
                    ],
                ):
                    result = main()

        assert result == 0
        kwargs = solver_cls.call_args.kwargs
        assert kwargs["mode"] == "cegis"
        assert kwargs["bin_solver_name"] == "cvc5"
        assert kwargs["num_workers"] == 8
        assert kwargs["num_samples"] == 6
        assert kwargs["sample_max_tries"] == 40
        assert kwargs["sample_seed_low"] == 3
        assert kwargs["sample_seed_high"] == 99
        assert kwargs["sample_config"] == {
            "lex_order": ["x", "y"],
            "jitter_real_delta": "0.25",
        }


class TestCLIArgumentValidation:
    """Tests for CLI argument validation."""

    def test_invalid_parser_choice(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--parser", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_theory_choice(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--theory", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_engine_choice(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--engine", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_log_level(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_BOOL_SAMPLE)
        with patch.object(
            sys, "argv", ["efsmt", str(in_file), "--log-level", "INVALID"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_eflira_forall_mode(self, tmp_path):
        in_file = _write_file(tmp_path, EFSMT_LIRA_SAMPLE)
        with patch.object(
            sys,
            "argv",
            [
                "efsmt",
                str(in_file),
                "--eflira-forall-mode",
                "invalid",
            ],
        ):
            with pytest.raises(SystemExit):
                main()
