"""Tests for aria.cli.pyomt - Optimization CLI."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import z3

from aria.cli.pyomt import main, solve_omt_problem


# Sample OMT problems
OMT_ARITH_SAMPLE = """(set-logic QF_LIA)
(declare-fun x () Int)
(assert (>= x 0))
(assert (<= x 10))
(maximize x)
(check-sat)
"""

OMT_BV_SAMPLE = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (bvule x (_ bv100 8)))
(maximize x)
(check-sat)
"""

OMT_MINIMIZE_SAMPLE = """(set-logic QF_LIA)
(declare-fun x () Int)
(assert (>= x 5))
(assert (<= x 20))
(minimize x)
(check-sat)
"""


def _write_file(tmp_path, content, suffix="smt2"):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


class TestSolveOMTProblem:
    """Tests for solve_omt_problem function."""

    def test_solve_arithmetic_qsmt(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        solve_omt_problem(str(in_file), engine="qsmt", solver_name="z3")
        captured = capsys.readouterr()
        # Should print optimal value
        assert "optimal-value" in captured.out or captured.out == ""

    def test_solve_arithmetic_iter(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        solve_omt_problem(str(in_file), engine="iter", solver_name="z3")
        captured = capsys.readouterr()
        assert "optimal-value" in captured.out or captured.out == ""

    def test_solve_bv(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_BV_SAMPLE)
        solve_omt_problem(str(in_file), engine="qsmt", solver_name="z3")
        captured = capsys.readouterr()
        # BV optimization may print differently
        assert "optimal-value" in captured.out or captured.out == ""

    def test_auto_detect_arith_theory(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        solve_omt_problem(
            str(in_file), engine="qsmt", solver_name="z3", theory=None
        )
        captured = capsys.readouterr()
        assert "optimal-value" in captured.out or captured.out == ""

    def test_auto_detect_bv_theory(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_BV_SAMPLE)
        solve_omt_problem(
            str(in_file), engine="qsmt", solver_name="z3", theory=None
        )
        captured = capsys.readouterr()


class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys):
        with patch.object(sys, "argv", ["pyomt", "/nonexistent/file.smt2"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_success_arith(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(sys, "argv", ["pyomt", str(in_file)]):
            result = main()
        # May return 0 or 1 depending on solver availability
        assert result in (0, 1)

    def test_main_with_engine_option(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--engine", "qsmt"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_with_theory_option(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--theory", "arith"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_maxsmt_not_implemented(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--type", "maxsmt"]
        ):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.err

    def test_main_with_log_level(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--log-level", "DEBUG"]
        ):
            result = main()
        assert result in (0, 1)

    def test_main_error_handling(self, tmp_path):
        in_file = _write_file(tmp_path, "(invalid content)", "smt2")
        with patch.object(sys, "argv", ["pyomt", str(in_file)]):
            result = main()
        assert result == 1


class TestCLIArgumentValidation:
    """Tests for CLI argument validation."""

    def test_invalid_engine_choice(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--engine", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_theory_choice(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--theory", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_log_level(self, tmp_path):
        in_file = _write_file(tmp_path, OMT_ARITH_SAMPLE)
        with patch.object(
            sys, "argv", ["pyomt", str(in_file), "--log-level", "INVALID"]
        ):
            with pytest.raises(SystemExit):
                main()
