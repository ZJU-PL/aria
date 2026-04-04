from aria.datalog import pyDatalog


def test_pydatalog_recursive_query():
    pyDatalog.clear()
    pyDatalog.assert_fact("parent", "bill", "john")
    pyDatalog.assert_fact("parent", "john", "sam")
    pyDatalog.load(
        """
        ancestor(X, Y) <= parent(X, Y)
        ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
        """
    )

    answers = pyDatalog.ask("ancestor('bill', Y)")

    assert ("john",) in answers.answers
    assert ("sam",) in answers.answers
