"""Tests for Boolean QBF parsing and lightweight solving support."""

from aria.bool.qbf import (
    PaserQCIR,
    PaserQDIMACS,
    QBF,
    QDIMACSParser,
    QCIRParser,
    TypedQDIMACSParser,
    parse_qcir_string,
    parse_qdimacs_string,
)


def test_parse_qdimacs_string_merges_adjacent_quantifier_blocks() -> None:
    parsed = parse_qdimacs_string(
        """
        p cnf 3 2
        e 1 0
        e 2 0
        a 3 0
        1 2 0
        -3 1 0
        """
    )

    assert parsed.parsed_prefix == [("e", [1, 2]), ("a", [3])]
    assert parsed.clauses == [[1, 2], [-3, 1]]


def test_qbf_solver_parses_and_solves_simple_existential_formula() -> None:
    parser = QDIMACSParser()
    qbf = parser.parse_qdimacs(
        """
        p cnf 1 1
        e 1 0
        1 0
        """
    )

    assert isinstance(qbf, QBF)
    assert parser.num_vars == 1
    assert parser.num_clauses == 1
    assert str(qbf.solve()) == "sat"
    assert qbf.quantifier_prefix_summary()["exists_blocks"] == 1


def test_qbf_solver_handles_simple_universal_unsat_formula() -> None:
    parser = QDIMACSParser()
    qbf = parser.parse_qdimacs(
        """
        p cnf 1 1
        a 1 0
        1 0
        """
    )

    assert str(qbf.solve()) == "unsat"


def test_compatibility_parsers_keep_old_names_and_flip_helpers() -> None:
    qdimacs = PaserQDIMACS.__name__
    qcir = QCIRParser.__name__
    assert qdimacs == "PaserQDIMACS"
    assert qcir == "PaserQCIR"

    typed = TypedQDIMACSParser("aria/tests/data/qbf_simple.qdimacs")
    flipped = typed.flip_and_assume(1, [2], [[-2, 1]])
    assert "p cnf 2 3" in flipped
    assert "2 0" in flipped


def test_parse_qcir_string_and_flip_assume() -> None:
    instance = parse_qcir_string(
        """
        #QCIR-G14
        exists(1)
        forall(2)
        output(3)
        3 = or(1,-2)
        """
    )
    assert instance.parsed_prefix == [("e", [1]), ("a", [2])]
    parser = PaserQCIR("aria/tests/data/qbf_simple.qcir")
    rewritten = parser.flip_and_assume(1, [1], [[-1, 2]])
    assert "output(" in rewritten
    assert "and(" in rewritten
