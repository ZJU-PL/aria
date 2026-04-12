from typing import cast

import pytest

from aria.datalog import DatalogParseError, Program, Relation, UndefinedPredicateError
from aria.datalog import util as datalog_util


def test_souffle_eqrel_reachable_port_queries_connected_components():
    """Port of tests/evaluation/eqrel_reachable/eqrel_reachable.dl."""

    program = Program()
    edge = program.relation("edge", 2)
    node = program.relation("node", 1)
    reachable = program.relation("reachable", 2)
    X, Y, Z = program.vars("X Y Z")

    for left, right in (("a", "b"), ("b", "c"), ("d", "e")):
        program.fact(edge(left, right))
        program.fact(edge(right, left))

    program.rule_of(node(X), edge(X, Y))
    program.rule_of(node(X), edge(Y, X))
    program.rule_of(reachable(X, X), node(X))
    program.rule_of(reachable(X, Y), reachable(X, Z), edge(Z, Y))

    assert sorted(program.rows(reachable("a", Y))) == [
        ("a",),
        ("b",),
        ("c",),
    ]
    assert sorted(program.rows(reachable("d", Y))) == [("d",), ("e",)]


def test_souffle_aggregate1_port_preserves_empty_aggregate_branch_behavior():
    """Port of tests/semantic/aggregate1/aggregate1.dl."""

    program = Program()
    a_rel = program.relation("A", 1)
    b_rel = program.relation("B", 1)
    total = program.function("total", 1)
    X, Y = program.vars("X Y")

    program.fact(a_rel(1))
    program.define_of(total("from_a") == program.agg.sum(X, for_each=X), a_rel(X))
    program.define_of(total("from_b") == program.agg.sum(X, for_each=X), b_rel(X))

    assert program.one_value(total("from_a") == Y) == 1
    assert program.rows(total("from_b") == Y) == []


def test_souffle_inline_negation_port_supports_negated_derived_predicates():
    """Port of tests/semantic/inline_negation/inline_negation.dl."""

    program = Program()
    c_rel = program.relation("c", 1)
    d_rel = program.relation("d", 2)
    b_rel = program.relation("b", 1)
    e_rel = program.relation("e", 1)
    a_rel = program.relation("a", 1)
    X, Y = program.vars("X Y")

    for value in (1, 2, 3, 4, 5):
        program.fact(c_rel(value))
    for left, right in ((1, 6), (1, 1), (2, 3), (4, 0), (1, 1)):
        program.fact(d_rel(left, right))

    program.rule_of(b_rel(X), d_rel(X, Y), ~c_rel(Y))
    program.rule_of(e_rel(X), d_rel(X, X), c_rel(X))
    program.rule_of(a_rel(X), c_rel(X), ~b_rel(X))
    program.rule_of(a_rel(X), c_rel(X), ~e_rel(X))

    b_values = sorted(cast(int, value) for value in set(program.scalar_rows(b_rel(X))))
    a_values = sorted(cast(int, value) for value in set(program.scalar_rows(a_rel(X))))

    assert b_values == [1, 4]
    assert program.scalar_rows(e_rel(X)) == [1]
    assert a_values == [2, 3, 4, 5]


def test_souffle_rule_undeclared_relation_port_raises_on_undeclared_query():
    """Adapted from tests/semantic/rule_undeclared_relation/rule_undeclared_relation.dl."""

    program = Program()
    X = program.var("X")
    declared = program.relation("b", 1)
    missing = Relation("missing", 1)

    program.fact(declared("value"))

    try:
        program.query(missing(X))
    except UndefinedPredicateError as exc:
        assert exc.predicate_name == "missing"
    else:
        raise AssertionError("Expected an explicit undefined predicate error.")


def test_souffle_empty_rule_port_reports_parse_location():
    """Adapted from tests/syntactic/rule/rule.dl empty-rule rejection."""

    program = Program()

    try:
        program.load("Path(A, C) <=")
    except DatalogParseError as exc:
        assert exc.line == 1
        assert exc.column is not None
        assert "Path(A, C) <=" in str(exc)
    else:
        raise AssertionError("Expected a parse error for an empty rule body.")


def test_souffle_aggregate5_port_keeps_filtered_minimum_selection():
    """Port of tests/semantic/aggregate5/aggregate5.dl."""

    program = Program()
    attribute = program.relation("attribute", 1)
    d_rel = program.relation("d", 2)
    sel = program.function("sel", 1)
    id_var, valid, minimum = program.vars("ID VALID MINIMUM")

    program.fact(attribute(1))
    program.fact(d_rel(2, 1))
    program.fact(d_rel(7, 0))

    program.define_of(
        sel("m") == program.agg.min(id_var, order_by=id_var),
        d_rel(id_var, valid),
        valid != 0,
        attribute(valid),
    )

    assert program.one_value(sel("m") == minimum) == 2


def test_souffle_complex_rule_port_splits_disjunction_into_equivalent_rules():
    """Port of tests/semantic/complex_rule/complex_rule.dl."""

    program = Program()
    a_rel = program.relation("a", 1)
    f_rel = program.relation("f", 1)
    query = program.relation("query", 1)
    X, Y = program.vars("X Y")

    program.fact(a_rel(1))
    program.fact(f_rel(1))

    program.rule_of(query(X), a_rel(X), f_rel(X))
    program.rule_of(query(X), a_rel(X), a_rel(Y))
    program.rule_of(query(X), a_rel(X), f_rel(Y))

    assert program.scalar_rows(query(X)) == [1]


def test_souffle_bin1_port_accepts_binary_integer_literals():
    """Port of tests/semantic/bin1/bin1.dl."""

    program = Program()
    binary = program.relation("Binary", 1)
    value = program.var("VALUE")

    program.facts(
        binary(0b0),
        binary(0b1),
        binary(0b11111111111111111111111111111111),
    )

    assert sorted(cast(int, item) for item in program.scalar_rows(binary(value))) == [
        0,
        1,
        4294967295,
    ]


def test_souffle_hex1_port_accepts_hexadecimal_integer_literals():
    """Port of tests/semantic/hex1/hex1.dl."""

    program = Program()
    hexadecimal = program.relation("Hexadecimal", 1)
    value = program.var("VALUE")

    program.facts(hexadecimal(0x0), hexadecimal(0x1), hexadecimal(0xFFFFFFFF))

    assert sorted(
        cast(int, item) for item in program.scalar_rows(hexadecimal(value))
    ) == [0, 1, 4294967295]


def test_souffle_multiple_heads_port_splits_shared_body_across_rules():
    """Adapted from tests/syntactic/multiple_heads/multiple_heads.dl."""

    program = Program()
    node = program.relation("node", 1)
    edge = program.relation("edge", 2)
    X, Y = program.vars("X Y")

    program.fact(node(1))
    program.fact(edge(1, 2))

    program.rule_of(edge(X, Y), edge(Y, X))
    program.rule_of(node(X), edge(X, Y))
    program.rule_of(node(Y), edge(X, Y))

    assert sorted(cast(int, item) for item in program.scalar_rows(node(X))) == [1, 2]
    assert sorted(program.rows(edge(X, Y))) == [(1, 2), (2, 1)]


def test_souffle_rel_stratification2_port_handles_arithmetic_recursion():
    """Port of tests/semantic/rel_stratification2/rel_stratification2.dl."""

    program = Program()
    input_rel = program.relation("Input", 1)
    relevant = program.relation("RelevantNumber", 1)
    trace = program.relation("Trace", 2)
    N, X, A, B = program.vars("N X A B")

    program.facts(input_rel(11), input_rel(12))
    program.rule_of(input_rel(X - 2), trace(X, X))
    program.rule_of(relevant(N), input_rel(N), input_rel(N + 1))
    program.rule_of(trace(A, B), relevant(A), relevant(B), ~input_rel(A + B))

    assert program.scalar_rows(relevant(N)) == [11]
    assert program.rows(trace(A, B)) == [(11, 11)]
    assert sorted(cast(int, item) for item in program.scalar_rows(input_rel(N))) == [
        11,
        12,
    ]


def test_souffle_aggregate2_port_counts_nonempty_and_empty_sources():
    """Adapted from tests/semantic/aggregate2/aggregate2.dl."""

    program = Program()
    a_rel = program.relation("A", 1)
    b_rel = program.relation("B", 1)
    total = program.function("total", 1)
    X, Y = program.vars("X Y")

    program.fact(a_rel("witness"))
    program.define_of(total("a") == program.agg.count(X), a_rel(X))
    program.define_of(total("b") == program.agg.count(X), b_rel(X))

    assert program.one_value(total("a") == Y) == 1
    assert program.rows(total("b") == Y) == []


def test_souffle_inline_ungrounded_port_reports_unbound_comparison():
    """Adapted from tests/semantic/inline_ungrounded/inline_ungrounded.dl."""

    program = Program()
    a_rel = program.relation("a", 2)
    b_rel = program.relation("b", 2)
    r0 = program.relation("r0", 1)
    X, Y = program.vars("X Y")

    program.rule_of(a_rel(X, Y), X == Y + 1)
    program.rule_of(b_rel(X, Y), 0 < X, 0 < Y)
    program.rule_of(r0(X), a_rel(X, 0))
    program.rule_of(r0(X), b_rel(X, 0))

    with pytest.raises(datalog_util.DatalogError, match="left hand side"):
        program.rows(r0(X))


def test_souffle_divide_by_zero_port_returns_no_results_without_crashing():
    """Adapted from tests/semantic/divide_by_zero/divide_by_zero.dl."""

    program = Program()
    test_divide = program.relation("test_divide", 1)
    X, Y = program.vars("X Y")

    program.rule_of(test_divide(X), X == 1, Y == 0, (X / Y) == 0, (X / 0) == 0)

    assert program.rows(test_divide(X)) == []


def test_souffle_identity_functor_port_preserves_python_literal_values():
    """Adapted from tests/semantic/identity_functor/identity_functor.dl."""

    program = Program()
    f2f = program.relation("F2F", 1)
    i2i = program.relation("I2I", 1)
    s2s = program.relation("S2S", 1)
    mk_unsigned = program.relation("mkUnsigned", 1)
    u2u = program.relation("U2U", 1)
    U = program.var("U")

    program.fact(f2f(float(3.0)))
    program.fact(i2i(int(3)))
    program.fact(s2s(str("hi")))
    program.fact(mk_unsigned(3))
    program.rule_of(u2u(U), mk_unsigned(U))

    assert program.scalar_rows(f2f(U)) == [3.0]
    assert program.scalar_rows(i2i(U)) == [3]
    assert program.scalar_rows(s2s(U)) == ["hi"]
    assert program.scalar_rows(u2u(U)) == [3]


def test_souffle_facts_port_filters_duplicate_facts():
    """Adapted from tests/evaluation/facts/facts.dl."""

    program = Program()
    relation = program.relation("N", 1)
    value = program.var("VALUE")

    program.fact(relation("0"))
    program.fact(relation("0"))
    program.fact(relation("1"))

    assert sorted(cast(str, item) for item in program.scalar_rows(relation(value))) == [
        "0",
        "1",
    ]


def test_souffle_facts2_port_evaluates_symbol_constants_before_assertion():
    """Adapted from tests/evaluation/facts2/facts2.dl."""

    program = Program()
    relation = program.relation("A", 1)
    value = program.var("VALUE")

    program.facts(relation("meow " + "meow"), relation(str(10)), relation(str(-10)))

    assert sorted(cast(str, item) for item in program.scalar_rows(relation(value))) == [
        "-10",
        "10",
        "meow meow",
    ]


def test_souffle_number_constants_port_accepts_signed_integer_extremes():
    """Port of tests/evaluation/number_constants/number_constants.dl."""

    program = Program()
    relation = program.relation("R", 1)
    value = program.var("VALUE")

    program.facts(
        relation(0),
        relation(1),
        relation(-1),
        relation(2147483647),
        relation(-2147483648),
    )

    assert sorted(cast(int, item) for item in program.scalar_rows(relation(value))) == [
        -2147483648,
        -1,
        0,
        1,
        2147483647,
    ]


def test_souffle_plus_port_supports_arithmetic_via_body_equalities():
    """Adapted from tests/evaluation/plus/plus.dl."""

    program = Program()
    relation = program.relation("R", 2)
    derived = program.relation("A", 3)
    X, Y, Z = program.vars("X Y Z")

    for left, right in ((1, 2), (2, 3), (3, 5)):
        program.fact(relation(left, right))

    program.rule_of(derived(X, Y, Z), relation(X, Y), Z == X + Y)

    assert sorted(program.rows(derived(X, Y, Z))) == [(1, 2, 3), (2, 3, 5), (3, 5, 8)]


def test_souffle_recursion_port_preserves_existing_facts_under_self_recursion():
    """Port of tests/evaluation/recursion/recursion.dl."""

    program = Program()
    relation = program.relation("p", 1)
    value = program.var("VALUE")

    program.facts(relation("0"), relation("1"))
    program.rule_of(relation(value), relation(value))

    assert sorted(cast(str, item) for item in program.scalar_rows(relation(value))) == [
        "0",
        "1",
    ]


def test_souffle_mutrecursion_port_closes_two_mutually_recursive_relations():
    """Port of tests/evaluation/mutrecursion/mutrecursion.dl."""

    program = Program()
    p_rel = program.relation("p", 1)
    q_rel = program.relation("q", 1)
    value = program.var("VALUE")

    program.facts(p_rel("a"), p_rel("b"), q_rel("c"), q_rel("d"))
    program.rule_of(p_rel(value), q_rel(value))
    program.rule_of(q_rel(value), p_rel(value))

    expected = ["a", "b", "c", "d"]
    assert sorted(cast(str, item) for item in program.scalar_rows(p_rel(value))) == expected
    assert sorted(cast(str, item) for item in program.scalar_rows(q_rel(value))) == expected


def test_souffle_trans_port_computes_transitive_closure():
    """Port of tests/example/trans/trans.dl."""

    program = Program()
    relation = program.relation("A", 2)
    X, Y, Z = program.vars("X Y Z")

    for left, right in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")):
        program.fact(relation(left, right))

    program.rule_of(relation(X, Z), relation(X, Y), relation(Y, Z))

    assert sorted(program.rows(relation("a", Z))) == [
        ("b",),
        ("c",),
        ("d",),
        ("e",),
    ]


def test_souffle_simple_port_filters_input_through_helper_relations():
    """Adapted from tests/evaluation/simple/simple.dl."""

    program = Program()
    the_input = program.relation("the_input", 1)
    the_output = program.relation("the_output", 1)
    hello_world = program.relation("hello_world", 1)
    is_hello_world = program.relation("is_hello_world", 1)
    X = program.var("X")

    program.facts(the_input("helloworld"), the_input("ignored"))
    program.fact(hello_world("helloworld"))
    program.rule_of(is_hello_world(X), the_input(X), hello_world(X))
    program.rule_of(the_output(X), is_hello_world(X))

    assert program.scalar_rows(the_output(X)) == ["helloworld"]


def test_souffle_negation_ports_match_reachability_filtering_cases():
    """Ports of tests/evaluation/neg1.dl through neg5.dl."""

    def compute(case: int):
        program = Program()
        x_rel = program.relation("X", 2)
        y_rel = program.relation("Y", 2)
        z_rel = program.relation("Z", 2)
        A, B, C = program.vars("A B C")

        for left, right in (("a", "b"), ("b", "c"), ("c", "d")):
            program.fact(x_rel(left, right))

        program.rule_of(y_rel(A, B), x_rel(A, B))
        program.rule_of(y_rel(A, C), x_rel(A, B), y_rel(B, C))

        if case == 1:
            program.rule_of(z_rel(A, B), ~x_rel(A, B), y_rel(A, B))
        elif case == 2:
            program.rule_of(z_rel(A, B), ~x_rel(A, B), y_rel(A, B), x_rel(A, B))
        elif case == 3:
            program.rule_of(z_rel(A, B), ~x_rel("a", B), y_rel(A, B))
        elif case == 4:
            program.rule_of(z_rel(A, B), ~x_rel("a", "b"), y_rel(A, B))
        else:
            program.rule_of(z_rel(A, B), ~x_rel("g", "h"), y_rel(A, B))

        return sorted(program.rows(z_rel(A, B)))

    assert compute(1) == []
    assert compute(2) == []
    assert compute(3) == []
    assert compute(4) == []
    assert compute(5) == [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ]


def test_souffle_neg6_port_handles_wildcard_negation_filters():
    """Port of tests/evaluation/neg6/neg6.dl."""

    program = Program()
    a_rel = program.relation("a", 3)
    b_rel = program.relation("b", 1)
    c_rel = program.relation("c", 1)
    X, Y, Z = program.vars("X Y Z")

    for row in ((1, 1, 1), (1, 2, 2), (1, 2, 3), (2, 1, 1), (3, 1, 1), (4, 2, 3)):
        program.fact(a_rel(*row))

    program.rule_of(b_rel(X), a_rel(X, Y, Z), ~a_rel(Y, X, Z))
    program.rule_of(c_rel(X), a_rel(X, Y, Z), ~a_rel(X, Y, X))

    expected = [1, 2, 3, 4]
    assert sorted(cast(int, item) for item in set(program.scalar_rows(b_rel(X)))) == expected
    assert sorted(cast(int, item) for item in set(program.scalar_rows(c_rel(X)))) == expected


def test_souffle_float_equality_port_treats_negative_zero_like_zero():
    """Adapted from tests/evaluation/float_equality/float_equality.dl."""

    program = Program()
    relation = program.relation("A", 1)
    X, Y = program.vars("X Y")

    program.fact(relation(0.0))
    program.rule_of(relation(Y), relation(X), X == -0.0, Y == X + 1)

    assert sorted(cast(float, item) for item in program.scalar_rows(relation(X))) == [0.0, 1.0]


def test_souffle_fib1_port_supports_recursive_numeric_derivations():
    """Adapted from tests/example/fib1/fib1.dl."""

    program = Program()
    fib = program.relation("fib", 2)
    N, X, Y, NEXT_N, VALUE = program.vars("N X Y NEXT_N VALUE")

    program.facts(fib(0, 0), fib(1, 1))
    program.rule_of(
        fib(NEXT_N, VALUE),
        fib(N, X),
        fib(N - 1, Y),
        N < 8,
        NEXT_N == N + 1,
        VALUE == X + Y,
    )

    assert sorted(program.rows(fib(N, VALUE))) == [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (7, 13),
        (8, 21),
    ]
