# coding: utf-8

from pathlib import Path

import pytest

import aria.cli.efmc_cli as efmc_cli


def test_parse_arguments_minimal(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "p.smt2"
    input_file.write_text("(set-logic HORN)\n(check-sat)\n", encoding="utf-8")

    monkeypatch.setattr(
        efmc_cli.sys,
        "argv",
        [
            "efmc",
            "--file",
            str(input_file),
        ],
    )

    args = efmc_cli.parse_arguments()
    assert args.file == str(input_file)
    assert args.lang == "auto"
    assert args.engine == "ef"
    assert args.log_level == "INFO"


@pytest.mark.parametrize(
    "ext,expected",
    [
        (".smt2", "verify_chc"),
        (".sy", "verify_sygus"),
        (".sl", "verify_sygus"),
        (".bpl", "verify_boogie"),
        (".c", "verify_c"),
    ],
)
def test_main_auto_dispatch_by_extension(monkeypatch, tmp_path: Path, ext: str, expected: str):
    input_file = tmp_path / f"p{ext}"
    input_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        efmc_cli.sys,
        "argv",
        [
            "efmc",
            "--file",
            str(input_file),
            "--lang",
            "auto",
        ],
    )

    calls = {"verify_chc": 0, "verify_sygus": 0, "verify_boogie": 0, "verify_c": 0}

    def record(name):
        def _fn(self, file_path):
            assert file_path == str(input_file)
            calls[name] += 1

        return _fn

    monkeypatch.setattr(efmc_cli.EFMCRunner, "verify_chc", record("verify_chc"))
    monkeypatch.setattr(efmc_cli.EFMCRunner, "verify_sygus", record("verify_sygus"))
    monkeypatch.setattr(efmc_cli.EFMCRunner, "verify_boogie", record("verify_boogie"))
    monkeypatch.setattr(efmc_cli.EFMCRunner, "verify_c", record("verify_c"))

    efmc_cli.main()
    assert calls[expected] == 1
    assert sum(calls.values()) == 1


def test_main_explicit_lang_dispatch(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "x.unknown"
    input_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        efmc_cli.sys,
        "argv",
        [
            "efmc",
            "--file",
            str(input_file),
            "--lang",
            "chc",
        ],
    )

    called = {}

    def fake_verify_chc(self, file_path):
        called["file_path"] = file_path

    monkeypatch.setattr(efmc_cli.EFMCRunner, "verify_chc", fake_verify_chc)
    efmc_cli.main()
    assert called["file_path"] == str(input_file)


def test_main_unsupported_extension_exits(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "x.unknown"
    input_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        efmc_cli.sys,
        "argv",
        [
            "efmc",
            "--file",
            str(input_file),
            "--lang",
            "auto",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        efmc_cli.main()
    assert excinfo.value.code == 1
