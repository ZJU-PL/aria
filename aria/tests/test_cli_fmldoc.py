from argparse import Namespace

from aria.cli.fmldoc import (
    handle_analyze,
    handle_formats,
    handle_translate,
    handle_validate,
)


DIMACS_SAMPLE = """c Simple CNF
p cnf 2 2
1 -2 0
2 0
"""


def _write_dimacs(tmp_path, content=DIMACS_SAMPLE, suffix="cnf"):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


def test_translate_dimacs_to_smtlib(tmp_path):
    in_file = _write_dimacs(tmp_path)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="dimacs",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    assert out_file.exists()
    assert "(set-logic" in out_file.read_text()
    assert "(assert" in out_file.read_text()


def test_validate_dimacs(tmp_path, capsys):
    in_file = _write_dimacs(tmp_path)
    args = Namespace(input_file=str(in_file), format=None)

    assert handle_validate(args) == 0
    captured = capsys.readouterr()
    assert "Successfully validated" in captured.out


def test_validate_missing_header(tmp_path, capsys):
    in_file = _write_dimacs(tmp_path, content="1 0\n2 0\n")
    args = Namespace(input_file=str(in_file), format="dimacs")

    assert handle_validate(args) == 1
    captured = capsys.readouterr()
    assert "Validation failed" in captured.err


def test_analyze_dimacs(tmp_path, capsys):
    in_file = _write_dimacs(tmp_path)
    args = Namespace(input_file=str(in_file), format=None)

    assert handle_analyze(args) == 0
    captured = capsys.readouterr()
    assert "Number of variables" in captured.out
    assert "Number of clauses" in captured.out


def test_formats_list():
    assert handle_formats(Namespace()) == 0
