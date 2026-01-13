"""Extended tests for Python frontend new features."""

from aria.symabs.ai_symabs.domains.interval import Interval, IntervalAbstractState, IntervalDomain
from aria.symabs.ai_symabs.frontend.python_program import PythonProgram


def test_break_statement():
    program = PythonProgram(
        """
x = 0
for i in range(10):
    x += 1
    if x >= 5:
        break
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
    assert interval_x.lower >= 5
    assert interval_x.upper <= 10


def test_continue_statement():
    program = PythonProgram(
        """
x = 0
for i in range(5):
    if i % 2 == 0:
        continue
    x += i
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
    # x = 1 + 3 = 4 (only odd indices: 1 and 3)
    assert interval_x.lower <= 4
    assert interval_x.upper >= 4


def test_elif_statement():
    program = PythonProgram(
        """
x = 0
if y < 0:
    x = -1
elif y == 0:
    x = 0
else:
    x = 1
    """
    )
    domain = IntervalDomain(["x", "y"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "y": Interval(-2, 2),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    # All branches assign a single value, so x should be [-1, 1]
    assert interval_x.lower == -1
    assert interval_x.upper == 1


def test_ternary_conditional():
    program = PythonProgram(
        """
x = y if y > 0 else -y
    """
    )
    domain = IntervalDomain(["x", "y"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "y": Interval(-5, 5),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    # x = max(y, -y) for y in [-5, 5], which is [0, 5]
    assert interval_x.lower == 0
    assert interval_x.upper == 5


def test_multiple_assignment():
    program = PythonProgram(
        """
a = b = c = 0
    """
    )
    domain = IntervalDomain(["a", "b", "c"])
    input_state = IntervalAbstractState(
        {
            "a": Interval(5, 10),
            "b": Interval(3, 8),
            "c": Interval(1, 4),
        }
    )

    output = program.transform(domain, input_state)
    assert output.interval_of("a") == Interval(0, 0)
    assert output.interval_of("b") == Interval(0, 0)
    assert output.interval_of("c") == Interval(0, 0)


def test_assert_statement():
    program = PythonProgram(
        """
x = 10
assert x >= 0
x = x + 1
    """
    )
    domain = IntervalDomain(["x"])
    input_state = IntervalAbstractState({"x": Interval(0, 0)})

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    assert interval_x.lower >= 10
    assert interval_x.upper >= 11


def test_break_in_nested_while():
    program = PythonProgram(
        """
x = 0
i = 0
while i < 10:
    j = 0
    while j < 5:
        x += 1
        if x >= 3:
            break
        j += 1
    if x >= 3:
        break
    i += 1
    """
    )
    domain = IntervalDomain(["x", "i", "j"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "i": Interval(0, 0),
            "j": Interval(0, 0),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    # x should be exactly 3 after the break
    assert interval_x.lower >= 3
    assert interval_x.upper <= 5


def test_elif_chain():
    program = PythonProgram(
        """
x = 0
if y < -5:
    x = -1
elif y < 0:
    x = 0
elif y == 0:
    x = 1
elif y < 5:
    x = 2
else:
    x = 3
    """
    )
    domain = IntervalDomain(["x", "y"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "y": Interval(-10, 10),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    # All branches assign a single value, so x should be [-1, 3]
    assert interval_x.lower == -1
    assert interval_x.upper == 3


def test_continue_in_nested_loops():
    program = PythonProgram(
        """
x = 0
for i in range(5):
    if i == 2:
        continue
    for j in range(3):
        if j == 1:
            continue
        x += 1
    """
    )
    domain = IntervalDomain(["x", "i", "j"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "i": Interval(0, 0),
            "j": Interval(0, 0),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    # i in [0,1,3,4] (skip 2), j in [0,2] (skip 1)
    # For each valid i, we add x += 1 twice (j=0 and j=2), so 2*4=8
    assert interval_x.lower <= 8
    assert interval_x.upper >= 8


def test_complex_ternary():
    program = PythonProgram(
        """
x = (y if y > z else z) if (y + z) > 0 else 0
    """
    )
    domain = IntervalDomain(["x", "y", "z"])
    input_state = IntervalAbstractState(
        {
            "x": Interval(0, 0),
            "y": Interval(-5, 5),
            "z": Interval(-3, 7),
        }
    )

    output = program.transform(domain, input_state)
    interval_x = output.interval_of("x")
    assert interval_x.lower <= 0
    assert interval_x.upper >= 7


def test_break_with_augmented_assignment():
    program = PythonProgram(
        """
x = 0
for i in range(100):
    x *= 2
    if x > 10:
        break
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
    # x = 2^k where k >= ceil(log2(10)) = 4, so x >= 16
    assert interval_x.lower >= 16
