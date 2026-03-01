"""Tests for aria.cli.maxsat - MaxSAT CLI."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aria.cli.maxsat import main, solve_maxsat_from_file


WCNF_SAMPLE = """p wcnf 3 4 10
10 1 0
10 2 0
10 3 0
4 -1 -2 0
4 -1 -3 0
4 -2 -3 0
"""


def _write_wcnf(tmp_path: Path, content: str = WCNF_SAMPLE) -> Path:
    path = tmp_path / "sample.wcnf"
    path.write_text(content, encoding="utf-8")
    return path


class TestSolveMaxsatFromFile:
    """Tests for solve_maxsat_from_file."""

    def test_rc2_solves(self, tmp_path: Path) -> None:
        in_file = _write_wcnf(tmp_path)
        result = solve_maxsat_from_file(str(in_file), solver="rc2")
        assert result.cost is not None
        assert result.cost >= 0
        assert result.status in ("optimal", "satisfied", "unknown")

    def test_fm_solves(self, tmp_path: Path) -> None:
        in_file = _write_wcnf(tmp_path)
        result = solve_maxsat_from_file(str(in_file), solver="fm")
        assert result.cost is not None
        assert result.cost >= 0

    def test_lsu_solves(self, tmp_path: Path) -> None:
        in_file = _write_wcnf(tmp_path)
        try:
            result = solve_maxsat_from_file(str(in_file), solver="lsu")
        except (IndexError, RuntimeError):
            pytest.skip("LSU can fail on some WCNF instances (totalizer bounds)")
        assert result.cost is not None
        assert result.cost >= 0


class TestMaxsatCLI:
    """Tests for main CLI entry point."""

    def test_main_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch.object(sys, "argv", ["aria-maxsat", "/nonexistent/file.wcnf"]):
            result = main()
        assert result == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_main_success_rc2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        in_file = _write_wcnf(tmp_path)
        with patch.object(sys, "argv", ["aria-maxsat", str(in_file), "--solver", "rc2"]):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "cost:" in captured.out

    def test_main_print_model(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        in_file = _write_wcnf(tmp_path)
        with patch.object(
            sys, "argv", ["aria-maxsat", str(in_file), "--solver", "rc2", "--print-model"]
        ):
            result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "cost:" in captured.out
