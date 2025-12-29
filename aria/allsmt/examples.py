"""
Examples of using the AllSMT API.

This module provides examples of using the AllSMT API with different solvers and formulas.
"""

from aria.allsmt import create_allsmt_solver


def z3_example() -> None:
    """Example of using the Z3-based AllSMT solver."""
    print("\n=== Z3 AllSMT Example ===")

    from z3 import Ints, Bools, And, Or  # pylint: disable=import-outside-toplevel

    # Create a Z3 solver
    solver = create_allsmt_solver("z3")

    # Define variables
    x, y = Ints('x y')
    a, b, c, d = Bools('a b c d')

    # Define constraints
    expr = And(
        a == (x + y > 0),
        c == ((2 * x + 3 * y) < -10),
        Or(a, b),
        Or(c, d)
    )

    # Solve the formula with a model limit
    print("Solving formula with Z3...")
    solver.solve(expr, [a, b, c, d], model_limit=20)

    # Print the models
    print("\nModels:")
    solver.print_models(verbose=True)

    # Print the model count
    print(f"\nTotal models: {solver.get_model_count()}")


def pysmt_example() -> None:
    """Example of using the PySMT-based AllSMT solver with Z3 expressions as input."""
    print("\n=== PySMT AllSMT Example (with Z3 input) ===")

    try:
        from z3 import Ints, Bools, And, Or  # pylint: disable=import-outside-toplevel

        # Create a PySMT solver
        solver = create_allsmt_solver("pysmt")

        # Define Z3 variables
        x, y = Ints('x y')
        a, b, c, d = Bools('a b c d')

        # Define Z3 constraints - same as in z3_example
        expr = And(
            a == (x + y > 0),
            c == ((2 * x + 3 * y) < -10),
            Or(a, b),
            Or(c, d)
        )

        # Solve the formula with a model limit
        print("Solving Z3 formula with PySMT...")
        solver.solve(expr, [a, b, c, d], model_limit=20)

        # Print the models
        print("\nModels:")
        solver.print_models(verbose=True)

        # Print the model count
        print(f"\nTotal models: {solver.get_model_count()}")
    except ImportError as e:
        print(f"PySMT example failed: {e}")


def mathsat_example() -> None:
    """Example of using the MathSAT-based AllSMT solver."""
    print("\n=== MathSAT AllSMT Example ===")

    try:
        from z3 import Ints, Bools, And, Or  # pylint: disable=import-outside-toplevel

        # Create a MathSAT solver
        solver = create_allsmt_solver("mathsat")

        # Define variables
        x, y = Ints('x y')
        a, b, c, d = Bools('a b c d')

        # Define constraints
        expr = And(
            a == (x + y > 0),
            c == ((2 * x + 3 * y) < -10),
            Or(a, b),
            Or(c, d)
        )

        # Solve the formula with a model limit
        print("Solving formula with MathSAT...")
        solver.solve(expr, [a, b, c, d], model_limit=20)

        # Print the models
        print("\nModels:")
        solver.print_models()

        # Print the model count
        print(f"\nTotal models: {solver.get_model_count()}")
    except (ImportError, AttributeError, RuntimeError) as e:
        print(f"MathSAT example failed: {e}")


def simple_example() -> None:
    """A simple example that works with any solver."""
    print("\n=== Simple Example (Default Solver) ===")

    from z3 import Ints, And  # pylint: disable=import-outside-toplevel

    # Create a solver (default is Z3)
    solver = create_allsmt_solver()

    # Define variables and constraints
    x, y = Ints('x y')
    expr = And(x + y == 5, x > 0, y > 0)

    # Solve the formula with a model limit
    print("Solving formula...")
    solver.solve(expr, [x, y], model_limit=10)

    # Print the models
    print("\nModels:")
    solver.print_models(verbose=True)

    # Print the model count
    print(f"\nTotal models: {solver.get_model_count()}")


def infinite_models_example() -> None:
    """Example with potentially infinite models."""
    print("\n=== Infinite Models Example ===")

    from z3 import Ints, Reals, And  # pylint: disable=import-outside-toplevel

    # Create a solver
    solver = create_allsmt_solver("z3")

    # Example 1: Integer solution with inequality
    print("\nExample 1: Integer solution with inequality")
    x, y = Ints('x y')
    expr = And(x > 0, y > 0, x + y > 10)

    print("Solving formula with potentially infinite models...")
    solver.solve(expr, [x, y], model_limit=5)
    solver.print_models(verbose=True)

    # Example 2: Real number solution
    print("\nExample 2: Real number solution")
    a, b = Reals('a b')
    expr = And(a > 0, b > 0, a + b == 1)

    print("Solving formula with uncountably infinite models...")
    solver.solve(expr, [a, b], model_limit=5)
    solver.print_models(verbose=True)


def run_all_examples() -> None:
    """Run all examples."""
    simple_example()
    z3_example()
    pysmt_example()
    mathsat_example()
    infinite_models_example()


if __name__ == "__main__":
    run_all_examples()
