"""Tests for Boolean QBF parsing and lightweight solving support."""

import pytest

from aria.bool.qbf import (
    PaserQCIR,
    PaserQDIMACS,
    QBF,
    QCIRFormulaParser,
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


def test_parse_qdimacs_string_normalizes_tautologies_duplicates_and_free_vars() -> None:
    parsed = parse_qdimacs_string(
        """
        p cnf 4 4
        e 1 0
        e 1 0
        a 2 0
        1 -1 3 0
        3 3 0
        3 0
        -2 1 0
        """
    )

    assert parsed.parsed_prefix == [("e", [3, 1]), ("a", [2])]
    assert parsed.clauses == [[3], [-2, 1]]
    assert parsed.num_clauses == 2
    assert parsed.num_vars == 3


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


def test_qcir_formula_parser_supports_free_xor_and_ite() -> None:
    parser = QCIRFormulaParser()
    qbf = parser.parse_qcir(
        """
        #QCIR-G14
        free(1)
        exists(2)
        output(5)
        3 = xor(1,2)
        5 = ite(1,3,-2)
        """
    )

    assert isinstance(qbf, QBF)
    assert str(qbf.solve()) == "sat"


def test_parse_qcir_string_tracks_explicit_and_inferred_free_variables() -> None:
    parsed = parse_qcir_string(
        """
        #QCIR-G14
        free(1)
        exists(2)
        output(4)
        4 = xor(1,2,3)
        """
    )

    assert parsed.parsed_prefix == [("f", [1, 3]), ("e", [2])]
    assert parsed.free_variables() == {1, 3}


def test_parse_qcir_string_rejects_cyclic_gate_graphs() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        parse_qcir_string(
            """
            #QCIR-G14
            exists(1)
            output(2)
            2 = and(3)
            3 = or(2)
            """
        )


def test_parse_qdimacs_string_rejects_duplicate_quantification() -> None:
    with pytest.raises(ValueError, match="quantified multiple times"):
        parse_qdimacs_string(
            """
            p cnf 2 1
            e 1 0
            a 1 0
            1 0
            """,
            normalize=False,
        )


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
