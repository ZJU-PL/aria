from aria.datalog import (
    DatalogAPIError,
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
