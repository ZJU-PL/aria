import pytest
import z3
from argparse import Namespace

from aria.cli.fmldoc_cli import (
    handle_analyze,
    handle_formats,
    handle_translate,
    handle_validate,
)
from aria.translator import cnf2smt


DIMACS_SAMPLE = """c Simple CNF
p cnf 2 2
1 -2 0
2 0
"""

QDIMACS_SAMPLE = """p cnf 1 1
e 1 0
1 0
"""

QCIR_SAMPLE = """#QCIR-G14
exists(1)
output(2)
2 = and(1)
"""

QCIR_DIRECT_OUTPUT_SAMPLE = """#QCIR-G14
output(1)
"""

OPB_SAMPLE = """* #variable= 2 #constraint= 2
1 x1 + 1 x2 >= 1 ;
1 x1 + -1 x2 >= 0 ;
"""

WCNF_SAMPLE = """p wcnf 2 2 10
10 1 0
4 -1 2 0
"""

WCNF_UNSAT_SAMPLE = """p wcnf 1 1 5
5 0
"""

OPB_OBJECTIVE_SAMPLE = """* #variable= 2 #constraint= 1
min: 1 x1 + 2 x2 ;
1 x1 + 1 x2 >= 1 ;
"""

OPB_EXTENDED_SAMPLE = """* #variable= 3 #constraint= 4
max:
  x1 + 2 ~x2
;
x1 + ~x2 > 0 ;
x1 + x2 < 2 ;
x1 + x3 != 1 ;
2 x1 - x2 >= 0 ;
"""

OPB_SOFT_SAMPLE = """* #variable= 2 #constraint= 3
soft: 10 ;
[10] 1 x1 >= 1 ;
[4] 1 x2 >= 1 ;
[3] 1 ~x1 + 1 ~x2 >= 1 ;
"""

OPB_PRODUCT_SAMPLE = """* #variable= 3 #constraint= 2
min: 2 x1 x2 - x3 ;
1 x1 x2 + 1 x3 >= 1 ;
"""

OPB_ESCAPED_NAME_SAMPLE = """* #variable= 2 #constraint= 1
1 x[1] + 1 x-2 >= 1 ;
"""

SMTLIB_BOOL_SAMPLE = """(set-logic QF_UF)
(declare-const a Bool)
(declare-const b Bool)
(assert (or a b))
(assert (not a))
(check-sat)
"""

SMTLIB2_SAMPLE = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(assert (and (bvsge x (_ bv0 32)) (bvsle x (_ bv1 32))))
(check-sat)
"""

SYGUS_SAMPLE = """(set-logic BV)
(synth-fun f ((x (_ BitVec 8))) (_ BitVec 8)
  ((Start (_ BitVec 8) (x))))
(declare-var x (_ BitVec 8))
(constraint (= (f x) x))
(check-synth)
"""


def _write_dimacs(tmp_path, content=DIMACS_SAMPLE, suffix="cnf"):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


def _write_file(tmp_path, name, content):
    path = tmp_path / name
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

    original_solver = z3.Solver()
    original_solver.add(cnf2smt.dimacs_to_z3(str(in_file)))

    translated_solver = z3.Solver()
    translated_solver.add(z3.parse_smt2_file(str(out_file)))

    assert translated_solver.check() == original_solver.check()


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


def test_translate_dimacs_to_lp(tmp_path):
    in_file = _write_dimacs(tmp_path)
    out_file = tmp_path / "sample.lp"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="dimacs",
        output_format="lp",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    assert out_file.read_text(encoding="utf-8") == "p1 :- p2.\np2.\n"


def test_translate_qdimacs_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.qdimacs", QDIMACS_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="qdimacs",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat


def test_translate_qcir_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.qcir", QCIR_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="qcir",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat


def test_translate_qcir_direct_output_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.qcir", QCIR_DIRECT_OUTPUT_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="qcir",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat
    assert "(declare-const q1 Bool)" in out_file.read_text(encoding="utf-8")


def test_translate_opb_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat


def test_translate_opb_objective_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_OBJECTIVE_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.sat
    assert "(minimize" in out_file.read_text(encoding="utf-8")


def test_translate_extended_opb_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_EXTENDED_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.sat
    content = out_file.read_text(encoding="utf-8")
    assert "(maximize" in content
    assert "(<= (+ |x1| |x2|) 1)" in content
    assert "(not (= (+ |x1| |x3|) 1))" in content
    assert "(>= (+ |x1| (- 1 |x2|)) 1)" in content


def test_translate_soft_opb_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_SOFT_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.sat
    content = out_file.read_text(encoding="utf-8")
    assert "(assert (>= |x1| 1))" in content
    assert "(assert-soft (>= |x2| 1) :weight 4)" in content
    assert ":weight 3" in content


def test_translate_product_opb_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_PRODUCT_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.sat
    content = out_file.read_text(encoding="utf-8")
    assert "(set-logic QF_NIA)" in content
    assert "(* |x1| |x2|)" in content


def test_translate_opb_with_nontrivial_names(tmp_path):
    in_file = _write_file(tmp_path, "sample.opb", OPB_ESCAPED_NAME_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="opb",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat
    content = out_file.read_text(encoding="utf-8")
    assert "|x[1]|" in content
    assert "|x-2|" in content


def test_translate_wcnf_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.wcnf", WCNF_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="wcnf",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.sat


def test_translate_wcnf_unsat_hard_clause_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.wcnf", WCNF_UNSAT_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="wcnf",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    optimizer = z3.Optimize()
    optimizer.from_file(str(out_file))
    assert optimizer.check() == z3.unsat
    assert "(assert false)" in out_file.read_text(encoding="utf-8")


def test_translate_smtlib2_to_dimacs(tmp_path):
    in_file = _write_file(tmp_path, "sample.smt2", SMTLIB_BOOL_SAMPLE)
    out_file = tmp_path / "sample.cnf"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="smtlib2",
        output_format="dimacs",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    original_solver = z3.Solver()
    original_solver.add(z3.parse_smt2_file(str(in_file)))
    translated_solver = z3.Solver()
    translated_solver.add(cnf2smt.dimacs_to_z3(str(out_file)))
    assert translated_solver.check() == original_solver.check()


def test_translate_sygus_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.sy", SYGUS_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="sygus",
        output_format="smtlib2",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    content = out_file.read_text(encoding="utf-8")
    assert "(declare-fun f" in content
    assert "(assert (= (f x) x))" in content
    assert "(check-sat)" in content
    solver = z3.Solver()
    solver.add(z3.parse_smt2_file(str(out_file)))
    assert solver.check() == z3.sat


def test_translate_smtlib2_to_sympy(tmp_path):
    pytest.importorskip("pysmt")
    sympy = pytest.importorskip("sympy")
    in_file = _write_file(tmp_path, "sample.smt2", SMTLIB2_SAMPLE)
    out_file = tmp_path / "sample.sympy"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format="smtlib2",
        output_format="sympy",
        auto_detect=False,
    )

    assert handle_translate(args) == 0
    content = out_file.read_text(encoding="utf-8")
    assert "x >= 0" in content
    assert "x <= 1" in content
    x = sympy.symbols("x")
    expr = sympy.sympify(content.strip(), locals={"x": x})
    assert expr.subs({x: 0}) == sympy.true
    assert expr.subs({x: 3}) == sympy.false


def test_translate_auto_detect_qdimacs_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.qdimacs", QDIMACS_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format=None,
        output_format=None,
        auto_detect=True,
    )

    assert handle_translate(args) == 0
    assert out_file.exists()


def test_translate_auto_detect_qcir_to_smtlib2(tmp_path):
    in_file = _write_file(tmp_path, "sample.qcir", QCIR_SAMPLE)
    out_file = tmp_path / "sample.smt2"
    args = Namespace(
        input_file=str(in_file),
        output_file=str(out_file),
        input_format=None,
        output_format=None,
        auto_detect=True,
    )

    assert handle_translate(args) == 0
    assert out_file.exists()


def test_analyze_dimacs(tmp_path, capsys):
    in_file = _write_dimacs(tmp_path)
    args = Namespace(input_file=str(in_file), format=None)

    assert handle_analyze(args) == 0
    captured = capsys.readouterr()
    assert "Number of variables" in captured.out
    assert "Number of clauses" in captured.out


def test_formats_list(capsys):
    assert handle_formats(Namespace()) == 0
    captured = capsys.readouterr()
    assert "qcir -> smtlib2" in captured.out
    assert "opb -> smtlib2" in captured.out
    assert "wcnf -> smtlib2" in captured.out
    assert "smtlib2 -> dimacs" in captured.out
