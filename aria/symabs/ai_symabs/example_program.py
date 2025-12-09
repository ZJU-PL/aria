"""Demonstrates use of symbolic abstraction for analyzing straight-line code.
In a real program analyzer this would be combined with a fixedpoint computation
engine to handle loops.
"""
from frontend.program import Program
from domains.sign import Sign, SignAbstractState, SignDomain


def main() -> None:
    """Construct and analyze the example program.
    """
    program = Program("""
        x += 5
        x -= y
        y += 5
        y -= 3
        x -= 6
        z += 1
    """)

    domain = SignDomain(["x", "y", "z"])
    input_state = SignAbstractState({
        "x": Sign.Negative,
        "y": Sign.Positive,
        "z": Sign.Negative,
    })

    output_state = program.transform(domain, input_state)
    print(output_state)

    assert output_state.sign_of("x") == Sign.Negative
    assert output_state.sign_of("y") == Sign.Positive
    assert output_state.sign_of("z") == Sign.Top


if __name__ == "__main__":
    main()
