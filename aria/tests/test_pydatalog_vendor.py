from aria.datalog import py_datalog


def test_pydatalog_recursive_query():
    py_datalog.clear()
    py_datalog.assert_fact("parent", "bill", "john")
    py_datalog.assert_fact("parent", "john", "sam")
    py_datalog.load(
        """
        ancestor(X, Y) <= parent(X, Y)
        ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
        """
    )

    answers = py_datalog.ask("ancestor('bill', Y)")

    assert ("john",) in answers.answers
    assert ("sam",) in answers.answers
