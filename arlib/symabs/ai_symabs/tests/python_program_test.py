from arlib.symabs.ai_symabs.domains.interval import Interval, IntervalAbstractState
from arlib.symabs.ai_symabs.domains.interval import IntervalDomain
from arlib.symabs.ai_symabs.frontend.python_program import PythonProgram


def test_assignments_straight_line():
    program = PythonProgram(
        """
x = 1
x += 2
"""
    )
    domain = IntervalDomain(["x"])
    input_state = IntervalAbstractState({"x": Interval(0, 0)})

    output = program.transform(domain, input_state)
    assert output.interval_of("x") == Interval(3, 3)


def test_if_else_merges_branches():
    program = PythonProgram(
        """
x = 0
if y > 0:
    x = y
else:
    x = -y
"""
    )
    domain = IntervalDomain(["x", "y"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "y": Interval(-5, 10),
        }
    )

    output = program.transform(domain, input_state)
    interval = output.interval_of("x")
    assert interval.lower <= 0
    assert interval.upper == 10


def test_for_range_counts_iterations():
    program = PythonProgram(
        """
x = 0
for i in range(0, 3):
    x += 1
"""
    )
    domain = IntervalDomain(["x", "i"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "i": Interval(0, 0),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    assert interval_x.lower == 0
    assert 3 <= interval_x.upper <= 4
    assert output.interval_of("i").upper >= 3
