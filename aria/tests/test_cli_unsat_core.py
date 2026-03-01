"""Tests for aria.cli.unsat_core - UNSAT core CLI."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import z3

from aria.cli.unsat_core import (
    _constraints_from_smt2,
    _format_cores,
    main,
    run_unsat_core,
)
from aria.unsat_core.unsat_core import UnsatCoreResult


SMT2_UNSAT = """(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 1))
(assert (<= x 0))
(check-sat)
"""

SMT2_SAT = """(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(assert (<= x 1))
(check-sat)
"""


def _write_smt2(tmp_path: Path, content: str, name: str = "formula.smt2") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestConstraintsFromSmt2:
    """Tests for _constraints_from_smt2."""

    def test_load_assertions(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, SMT2_UNSAT)
        constraints = _constraints_from_smt2(str(p))
        assert len(constraints) == 2
        assert all(isinstance(c, z3.ExprRef) for c in constraints)


class TestRunUnsatCore:
    """Tests for run_unsat_core."""

    def test_unsat_returns_core(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, SMT2_UNSAT)
        constraints = _constraints_from_smt2(str(p))
        result = run_unsat_core(constraints, algorithm="marco")
        assert isinstance(result, UnsatCoreResult)
        assert len(result.cores) >= 1
        assert all(isinstance(c, set) for c in result.cores)

    def test_empty_constraints_raises(self) -> None:
        with pytest.raises(ValueError, match="No assertions"):
            run_unsat_core([])


class TestFormatCores:
    """Tests for _format_cores."""

    def test_formats_indices(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, SMT2_UNSAT)
        constraints = _constraints_from_smt2(str(p))
        result = run_unsat_core(constraints, algorithm="marco")
        out = _format_cores(result, constraints)
        assert "core 1" in out
        assert "indices" in out


class TestUnsatCoreCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["aria-unsat-core", "/nonexistent/file.smt2"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_sat_exits_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = _write_smt2(tmp_path, SMT2_SAT)
        with patch.object(sys, "argv", ["aria-unsat-core", str(p)]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "satisfiable" in captured.out

    def test_main_unsat_prints_core(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = _write_smt2(tmp_path, SMT2_UNSAT)
        with patch.object(sys, "argv", ["aria-unsat-core", str(p)]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "core" in captured.out or "indices" in captured.out
