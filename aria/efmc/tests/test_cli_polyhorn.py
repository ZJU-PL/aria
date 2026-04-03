# coding: utf-8

from pathlib import Path

import pytest

pytest.importorskip("numpy")

import aria.cli.polyhorn_cli as polyhorn_cli


def test_create_config_from_args_applies_overrides(monkeypatch):
    def fake_add_default_config(config):
        return {**config, "from_default": True}

    monkeypatch.setattr(polyhorn_cli, "add_default_config", fake_add_default_config)

    args = type(
        "Args",
        (),
        {
            "theorem": "farkas",
            "solver": "z3",
            "degree_sat": 1,
            "degree_unsat": 2,
            "degree_strict": 3,
            "integer": True,
            "sat_heuristic": True,
            "unsat_core": False,
        },
    )()

    config = polyhorn_cli.create_config_from_args(args)
    assert config["theorem_name"] == "farkas"
    assert config["solver_name"] == "z3"
    assert config["degree_of_sat"] == 1
    assert config["degree_of_nonstrict_unsat"] == 2
    assert config["degree_of_strict_unsat"] == 3
    assert config["integer_arithmetic"] is True
    assert config["SAT_heuristic"] is True
    assert config["unsat_core_heuristic"] is False
    assert config["from_default"] is True


def test_polyhorn_runner_writes_output_file(monkeypatch, tmp_path: Path):
    def fake_execute(file_path, config):
        assert file_path == str(tmp_path / "p.smt2")
        assert config["k"] == "v"
        return "sat", {"x": 1}

    monkeypatch.setattr(polyhorn_cli, "execute", fake_execute)

    runner = polyhorn_cli.PolyHornRunner()
    output_file = tmp_path / "out.json"
    input_file = tmp_path / "p.smt2"
    input_file.write_text("(set-logic QF_LRA)\n(check-sat)\n", encoding="utf-8")

    runner.solve_file(str(input_file), {"k": "v"}, output_file=str(output_file))
    assert output_file.exists()

    text = output_file.read_text(encoding="utf-8")
    assert '"result"' in text
    assert '"sat"' in text
    assert '"model"' in text


def test_main_exits_on_missing_input(monkeypatch, tmp_path: Path, capsys):
    missing = tmp_path / "missing.smt2"

    monkeypatch.setattr(
        polyhorn_cli.sys,
        "argv",
        [
            "polyhorn",
            str(missing),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        polyhorn_cli.main()
    assert excinfo.value.code == 1
    assert "does not exist" in capsys.readouterr().out
