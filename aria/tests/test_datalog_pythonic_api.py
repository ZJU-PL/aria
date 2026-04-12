from pathlib import Path

from aria.datalog import (
    DatalogAPIError,
    DatalogParseError,
    Program,
    Relation,
    UndefinedPredicateError,
    vars_,
)


def test_pythonic_program_recursive_query():
    program = Program()
    parent = program.relation("parent", 2)
    ancestor = program.relation("ancestor", 2)
    X, Y, Z = program.vars("X Y Z")

    program.fact(parent("bill", "john"))
    program.fact(parent("john", "sam"))
    program.rule(ancestor(X, Y)).when(parent(X, Y))
    program.rule(ancestor(X, Y)).when(parent(X, Z), ancestor(Z, Y))

    result = program.query(ancestor("bill", Y))

    assert result.rows() == [("john",), ("sam",)]
    assert list(result) == [("john",), ("sam",)]
    assert len(result) == 2
    assert result.scalar_rows() == ["john", "sam"]
    assert result.named_rows() == [{"Y": "john"}, {"Y": "sam"}]
    assert result.first() == ("john",)
    assert result.first_value() == "john"


def test_pythonic_query_supports_conjunctions():
    program = Program()
    salary = program.relation("salary", 2)
    manager = program.relation("manager", 2)
    X, Y = program.vars("X Y")

    program.fact(salary("sam", 5900))
    program.fact(salary("mary", 6300))
    program.fact(manager("sam", "mary"))

    result = program.query((salary(X, 5900)) & manager(X, Y))

    assert result.rows() == [("sam", "mary")]
    assert result.named_rows() == [{"X": "sam", "Y": "mary"}]


def test_program_convenience_helpers_cover_common_query_patterns():
    program = Program()
    parent = program.relation("parent", 2)
    ancestor = program.relation("ancestor", 2)
    X, Y, Z = program.vars("X Y Z")

    program.facts(parent("bill", "john"), parent("john", "sam"))
    program.rule_of(ancestor(X, Y), parent(X, Y))
    program.rule_of(ancestor(X, Y), parent(X, Z), ancestor(Z, Y))

    assert program.exists(ancestor("bill", Y))
    assert program.rows(ancestor("bill", Y)) == [("john",), ("sam",)]
    assert program.scalar_rows(ancestor("bill", Y)) == ["john", "sam"]
    assert program.first(ancestor("bill", Y)) == ("john",)
    assert program.first_value(ancestor("bill", Y)) == "john"
    assert program.one(parent("bill", Y)) == ("john",)
    assert program.one_value(parent("bill", Y)) == "john"


def test_program_supports_function_and_aggregate_builders():
    program = Program()
    manager = program.function("manager", 1)
    indirect_manager = program.relation("indirect_manager", 2)
    report_count = program.function("report_count", 1)
    budget = program.function("budget", 1)
    lowest = program.function("lowest", 1)
    salary = program.function("salary", 1)
    X, Y, Z, N = program.vars("X Y Z N")

    program.define_of(manager("sam") == "mary")
    program.define_of(manager("john") == "mary")
    program.define_of(salary("sam") == 5900)
    program.define_of(salary("john") == 6100)
    program.rule_of(indirect_manager(X, Y), manager(X) == Y)
    program.rule_of(indirect_manager(X, Y), manager(X) == Z, indirect_manager(Z, Y))
    program.define_of(
        report_count(X) == program.agg.count(Y),
        indirect_manager(Y, X),
    )
    program.define_of(
        budget(X) == program.agg.sum(N, for_each=Y),
        indirect_manager(Y, X),
        salary(Y) == N,
    )
    program.define_of(
        lowest(1) == program.agg.min(X, order_by=N),
        salary(X) == N,
    )

    assert program.one_value(manager("sam") == Y) == "mary"
    assert program.one_value(report_count("mary") == N) == 2
    assert program.one_value(budget("mary") == N) == 12000
    assert program.one_value(lowest(1) == X) == "sam"


def test_program_retract_all_handles_multiple_facts():
    program = Program()
    edge = program.relation("edge", 2)
    X = program.var("X")

    program.facts(edge("a", "b"), edge("a", "c"))
    assert sorted(program.scalar_rows(edge("a", X))) == ["b", "c"]

    program.retract_all(edge("a", "b"), edge("a", "c"))
    assert program.scalar_rows(edge("a", X)) == []


def test_pythonic_program_clear_resets_facts():
    program = Program()
    parent = program.relation("parent", 2)
    X = program.var("X")

    program.fact(parent("bill", "john"))
    assert program.query(parent("bill", X)).rows() == [("john",)]

    program.clear()

    assert program.query(parent("bill", X)).rows() == []


def test_query_on_undeclared_relation_raises_explicit_error():
    program = Program()
    X = program.var("X")
    orphan = Relation("orphan", 1)

    try:
        program.query(orphan(X))
    except UndefinedPredicateError as exc:
        assert "orphan" in str(exc)
    else:
        raise AssertionError("Expected explicit undefined predicate error.")


def test_pythonic_relation_arity_is_checked():
    program = Program()
    parent = program.relation("parent", 2)

    try:
        parent("bill")
    except ValueError as exc:
        assert "expects 2 arguments" in str(exc)
    else:
        raise AssertionError("Expected arity validation failure.")


def test_vars_helper_can_be_imported():
    X, Y = vars_("X Y")

    assert X.name == "X"
    assert Y.name == "Y"


def test_query_result_one_value_helpers():
    program = Program()
    parent = program.relation("parent", 2)
    X = program.var("X")

    program.fact(parent("bill", "john"))
    result = program.query(parent("bill", X))

    assert result.one() == ("john",)
    assert result.one_value() == "john"


def test_query_result_one_value_fails_for_multiple_rows():
    program = Program()
    parent = program.relation("parent", 2)
    X = program.var("X")

    program.fact(parent("bill", "john"))
    program.fact(parent("bill", "sam"))
    result = program.query(parent("bill", X))

    try:
        result.one_value()
    except DatalogAPIError as exc:
        assert "exactly one result row" in str(exc)
    else:
        raise AssertionError("Expected one_value() to reject multiple rows.")


def test_program_load_supports_string_rules_and_facts():
    program = Program()
    ancestor = program.relation("ancestor", 2)
    Y = program.var("Y")

    program.load(
        """
        + parent('bill', 'john')
        + parent('john', 'sam')
        ancestor(X, Y) <= parent(X, Y)
        ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
        """
    )

    result = program.query(ancestor("bill", Y))

    assert result.scalar_rows() == ["john", "sam"]


def test_program_load_registers_loaded_relations_for_querying():
    program = Program()
    X = program.var("X")

    program.load(
        """
        + edge('a', 'b')
        reachable(X, Y) <= edge(X, Y)
        """
    )

    reachable = program.relation("reachable", 2)
    assert program.query(reachable("a", X)).scalar_rows() == ["b"]


def test_program_load_file_supports_external_rule_files(tmp_path: Path):
    program = Program()
    Y = program.var("Y")
    path = tmp_path / "family.dl"
    path.write_text(
        """
        + parent('bill', 'john')
        + parent('john', 'sam')
        ancestor(X, Y) <= parent(X, Y)
        ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
        """,
        encoding="utf-8",
    )

    loaded_path = program.load_file(path)

    ancestor = program.relation("ancestor", 2)
    assert loaded_path == path
    assert program.query(ancestor("bill", Y)).scalar_rows() == ["john", "sam"]


def test_program_ask_supports_textual_queries():
    program = Program()
    program.load(
        """
        + parent('bill', 'john')
        + parent('john', 'sam')
        ancestor(X, Y) <= parent(X, Y)
        ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
        """
    )

    result = program.ask("ancestor('bill', Y)")

    assert result.scalar_rows() == ["john", "sam"]
    assert result.named_rows() == [{"Y": "john"}, {"Y": "sam"}]


def test_program_load_reports_line_aware_parse_errors():
    program = Program()

    try:
        program.load(
            """
            + parent('bill', 'john')
            ancestor(X, Y) <= parent(X, Y
            """
        )
    except DatalogParseError as exc:
        assert exc.line == 2
        assert exc.column is not None
        assert "ancestor(X, Y) <= parent(X, Y" in str(exc)
    else:
        raise AssertionError("Expected a parse error with source context.")


def test_program_ask_reports_line_aware_parse_errors():
    program = Program()

    try:
        program.ask("ancestor('bill', Y")
    except DatalogParseError as exc:
        assert exc.line == 1
        assert exc.column is not None
        assert "ancestor('bill', Y" in str(exc)
    else:
        raise AssertionError("Expected a parse error with source context.")
