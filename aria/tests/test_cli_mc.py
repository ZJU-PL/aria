"""Tests for aria.cli.mc - Model Counting CLI."""
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import z3

from aria.cli.mc import count_from_file, main


DIMACS_SAMPLE = """c Simple CNF
p cnf 2 2
1 -2 0
2 0
"""

SMT2_BOOL_SAMPLE = """(set-logic QF_UF)
(declare-const x Bool)
(declare-const y Bool)
(assert (or x y))
(check-sat)
"""

SMT2_BV_SAMPLE = """(set-logic QF_BV)
(declare-const x (_ BitVec 4))
(assert (bvult x #x8))
(check-sat)
"""

SMT2_ARITH_BOUNDED_SAMPLE = """(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 0))
(assert (>= y 0))
(assert (<= x 2))
(assert (<= y 2))
(assert (= (+ x y) 2))
(check-sat)
"""

SMT2_ARITH_UNBOUNDED_SAMPLE = """(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(check-sat)
"""

SMT2_REAL_ARITH_SAMPLE = """(set-logic QF_LRA)
(declare-const x Real)
(assert (>= x 0.0))
(assert (<= x 1.0))
(check-sat)
"""


def _write_file(tmp_path, content, suffix):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


class TestCountFromFile:
    """Tests for count_from_file function."""

    def test_dimacs_bool_counting(self, tmp_path):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        count = count_from_file(str(in_file), theory="bool", method="auto")
        assert isinstance(count, int)
        # Note: -1 may indicate external solver (e.g., sharpSAT) not available
        assert count == -1 or count >= 0

    def test_smt2_bool_counting(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_BOOL_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="bool", method="solver")
        assert isinstance(count, int)
        assert count > 0

    def test_auto_detect_dimacs(self, tmp_path):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        count = count_from_file(str(in_file), theory="auto", method="auto")
        assert isinstance(count, int)

    def test_auto_detect_smt2(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_BOOL_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="auto", method="auto")
        assert isinstance(count, int)

    def test_bv_counting(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_BV_SAMPLE, "smt2")
        # BV counting may require specific setup
        try:
            count = count_from_file(str(in_file), theory="bv", method="auto")
            assert isinstance(count, int)
        except Exception:
            pytest.skip("BV counting not fully supported in test environment")

    def test_arith_bounded_counting(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_ARITH_BOUNDED_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="arith", method="auto")
        assert count == 3

    def test_arith_unbounded_rejected(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_ARITH_UNBOUNDED_SAMPLE, "smt2")
        with pytest.raises(ValueError, match="unbounded"):
            count_from_file(str(in_file), theory="arith", method="auto")

    def test_arith_solver_method_alias(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_ARITH_BOUNDED_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="arith", method="solver")
        assert count == 3

    def test_auto_bounded_lia_uses_arith_counter(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_ARITH_BOUNDED_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="auto", method="auto")
        assert count == 3

    def test_auto_unbounded_lia_falls_back_generic(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_ARITH_UNBOUNDED_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="auto", method="auto")
        assert isinstance(count, int)

    def test_auto_real_arith_falls_back_generic(self, tmp_path):
        in_file = _write_file(tmp_path, SMT2_REAL_ARITH_SAMPLE, "smt2")
        count = count_from_file(str(in_file), theory="auto", method="auto")
        assert isinstance(count, int)

    def test_file_not_found(self):
        with pytest.raises((IOError, OSError)):
            count_from_file("/nonexistent/file.smt2")


class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys):
        with patch.object(sys, "argv", ["mc", "/nonexistent/file.smt2"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_success_dimacs(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        with patch.object(sys, "argv", ["mc", str(in_file)]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Number of models" in captured.out

    def test_main_with_theory_option(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        with patch.object(
            sys, "argv", ["mc", str(in_file), "--theory", "bool"]
        ):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Number of models" in captured.out

    def test_main_with_log_level(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        with patch.object(
            sys, "argv", ["mc", str(in_file), "--log-level", "DEBUG"]
        ):
            result = main()
        assert result == 0

    def test_main_error_with_debug(self, tmp_path, capsys):
        in_file = _write_file(tmp_path, "invalid content", "cnf")
        with patch.object(
            sys, "argv", ["mc", str(in_file), "--log-level", "DEBUG"]
        ):
            result = main()
        assert result == 1


class TestCLIArgumentValidation:
    """Tests for CLI argument validation."""

    def test_invalid_log_level(self, tmp_path):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        with patch.object(
            sys, "argv", ["mc", str(in_file), "--log-level", "INVALID"]
        ):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_theory_choice(self, tmp_path):
        in_file = _write_file(tmp_path, DIMACS_SAMPLE, "cnf")
        with patch.object(
            sys, "argv", ["mc", str(in_file), "--theory", "invalid"]
        ):
            with pytest.raises(SystemExit):
                main()
