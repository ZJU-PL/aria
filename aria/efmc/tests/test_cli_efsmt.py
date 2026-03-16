# coding: utf-8

import builtins
from pathlib import Path

import pytest

import aria.efmc.cli.efsmt as efsmt_cli


def test_parse_arguments_defaults(monkeypatch, tmp_path: Path):
    query_file = tmp_path / "q.smt2"
    query_file.write_text("(set-logic QF_BV)\n(check-sat)\n", encoding="utf-8")

    monkeypatch.setattr(
        efsmt_cli.sys,
        "argv",
        [
            "efsmt",
            "--file",
            str(query_file),
            "z3",
        ],
    )

    args = efsmt_cli.parse_arguments()
    assert args.file == str(query_file)
    assert args.solver == "z3"
    assert args.logic == "BV"
    assert args.parallel is False
    assert args.timeout == 30
    assert args.dump_smt2 is False
    assert args.dump_qbf is False
    assert args.dump_cnf is False


def test_main_falls_back_when_pysmt_missing(monkeypatch, tmp_path: Path, capsys):
    query_file = tmp_path / "q.smt2"
    query_file.write_text("(set-logic QF_BV)\n(check-sat)\n", encoding="utf-8")

    monkeypatch.setattr(
        efsmt_cli.sys,
        "argv",
        [
            "efsmt",
            "--file",
            str(query_file),
            "z3",
            "--parallel",
        ],
    )
    monkeypatch.setattr(efsmt_cli, "PYSMT_AVAILABLE", False)

    called = {}

    def fake_solve(
        self,
        file_name,
        solver_name,
        logic,
        parallel,
        timeout,
        dump_smt2,
        dump_qbf,
        dump_cnf,
    ):
        called["kwargs"] = {
            "file_name": file_name,
            "solver_name": solver_name,
            "logic": logic,
            "parallel": parallel,
            "timeout": timeout,
            "dump_smt2": dump_smt2,
            "dump_qbf": dump_qbf,
            "dump_cnf": dump_cnf,
        }
        return None

    monkeypatch.setattr(efsmt_cli.EFSMTRunner, "solve_efsmt_file", fake_solve)

    efsmt_cli.main()
    out = capsys.readouterr().out
    assert "pySMT not available" in out
    assert called["kwargs"]["parallel"] is False


def test_signal_handler_exits(monkeypatch):
    def fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(efsmt_cli.sys, "exit", fake_exit)
    with pytest.raises(SystemExit) as excinfo:
        efsmt_cli.signal_handler(None, None)
    assert excinfo.value.code == 0

