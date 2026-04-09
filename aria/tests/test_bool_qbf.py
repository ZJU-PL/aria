"""Tests for Boolean-level QBF parsing helpers."""

import z3

from aria.bool.analysis import quantifier_prefix_report
from aria.bool.qbf import (
    QCIRFormulaParser,
    QDIMACSParser,
    parse_qcir_string,
    parse_qdimacs_string,
)
from aria.translator.parsing import parse_qdimacs_file


QDIMACS_SAMPLE = """c sample
p cnf 3 2
a 1 0
e 2 3 0
1 -2 0
3 0
"""


QCIR_SAMPLE = """#QCIR-G14
forall(1)
exists(2,3)
output(4)
4 = and(1,2,-3)
"""


def test_parse_qdimacs_string_reports_prefix_and_clause_data():
    parsed = parse_qdimacs_string(QDIMACS_SAMPLE)

    assert parsed.num_vars == 3
    assert parsed.num_clauses == 2
    assert parsed.parsed_prefix == [("a", [1]), ("e", [2, 3])]
    assert parsed.clauses == [[1, -2], [3]]
    assert quantifier_prefix_report(parsed)["prefix_pattern"] == "ae"


def test_qdimacs_parser_builds_satisfiable_qbf():
    parser = QDIMACSParser()
    qbf = parser.parse_qdimacs(QDIMACS_SAMPLE)

    assert qbf.quantifier_depth() == 2
    assert qbf.quantifier_prefix_summary()["forall_vars"] == 1
    assert qbf.solve() == z3.sat
    assert qbf.solve(backend="pysat") == z3.sat


def test_parse_qcir_string_preserves_gate_shape():
    parsed = parse_qcir_string(QCIR_SAMPLE)

    assert parsed.parsed_prefix == [("a", [1]), ("e", [2, 3])]
    assert parsed.output_gate == 4
    assert parsed.parsed_gates == [("and", "4", ["1", "2", "-3"])]
    assert quantifier_prefix_report(parsed)["alternations"] == 1


def test_qcir_formula_parser_builds_sat_formula_from_gate_circuit():
    parser = QCIRFormulaParser()
    qbf = parser.parse_qcir(
        """
        #QCIR-G14
        forall(1)
        exists(2)
        output(4)
        3 = xor(1,2)
        4 = or(3,-2)
        """
    )

    assert qbf.quantifier_depth() == 2
    assert qbf.quantifier_prefix_summary()["forall_vars"] == 1
    assert qbf.solve() == z3.sat


def test_legacy_qdimacs_file_parser_returns_compat_wrapper(tmp_path):
    path = tmp_path / "sample.qdimacs"
    path.write_text(QDIMACS_SAMPLE, encoding="utf-8")

    parsed = parse_qdimacs_file(str(path))

    assert parsed.preamble == ["p", "cnf", "3", "2"]
    assert parsed.parsed_prefix == [("a", [1]), ("e", [2, 3])]
    assert parsed.clauses == ["1 -2 0", "3 0"]


def test_pysat_qbf_solver_handles_true_and_false_prefixes():
    parser = QDIMACSParser()
    true_qbf = parser.parse_qdimacs(
        """
        p cnf 2 1
        a 1 0
        e 2 0
        1 2 0
        """
    )
    false_qbf = parser.parse_qdimacs(
        """
        p cnf 2 2
        a 1 0
        e 2 0
        1 0
        2 0
        """
    )

    assert true_qbf.solve(backend="pysat") == z3.sat
    assert false_qbf.solve(backend="pysat") == z3.unsat
