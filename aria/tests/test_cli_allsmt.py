"""Tests for aria.cli.allsmt_cli - AllSMT CLI."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aria.cli.allsmt_cli import _formula_and_vars_from_smt2, enumerate_models, main


SMT2_SAT_TWO_MODELS = """(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)
(assert (>= x 0))
(assert (<= x 1))
(assert (>= y 0))
(assert (<= y 1))
(assert (or (= x 0) (= x 1)))
(check-sat)
"""


def _write_smt2(tmp_path: Path, content: str, name: str = "formula.smt2") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestFormulaAndVarsFromSmt2:
    """Tests for _formula_and_vars_from_smt2."""

    def test_load_formula_and_vars(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, SMT2_SAT_TWO_MODELS)
        formula, variables = _formula_and_vars_from_smt2(str(p))
        assert formula is not None
        assert len(variables) >= 1
        names = {v.decl().name() for v in variables}
        assert "x" in names or "y" in names

    def test_empty_assertions_raises(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, "(set-logic QF_LIA)\n(declare-const x Int)\n(check-sat)\n")
        with pytest.raises(ValueError, match="No assertions"):
            _formula_and_vars_from_smt2(str(p))


class TestEnumerateModels:
    """Tests for enumerate_models."""

    def test_enumerates_models(self, tmp_path: Path) -> None:
        p = _write_smt2(tmp_path, SMT2_SAT_TWO_MODELS)
        count = enumerate_models(str(p), solver_name="z3", model_limit=10)
        assert count >= 1
        assert count <= 10


class TestAllsmtCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["aria-allsmt", "/nonexistent/file.smt2"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_count_only(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = _write_smt2(tmp_path, SMT2_SAT_TWO_MODELS)
        with patch.object(sys, "argv", ["aria-allsmt", str(p), "--count-only", "--limit", "5"]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.strip().isdigit() or "Model" in captured.out

    def test_main_prints_models(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = _write_smt2(tmp_path, SMT2_SAT_TWO_MODELS)
        with patch.object(sys, "argv", ["aria-allsmt", str(p), "--limit", "3"]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Model" in captured.out or captured.out.strip().isdigit()
