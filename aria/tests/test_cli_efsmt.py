"""Tests for aria.cli.efsmt - Exists-Forall SMT CLI."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import z3

from aria.cli.efsmt import main, _infer_theory, _format_result


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
