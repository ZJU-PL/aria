"""
Examples of using the sampling API.

This module provides examples of using the sampling API with different logics and methods.
"""

import z3
from aria.sampling import (
    sample_models_from_formula, create_sampler,
    Logic, SamplingMethod, SamplingOptions, SamplingResult
)


def boolean_example():
    """Example of sampling from a Boolean formula."""
    print("\n=== Boolean Sampling Example ===")

    # Create a Boolean formula
    a, b, c = z3.Bools('a b c')
    formula = z3.And(z3.Or(a, b), z3.Or(b, c), z3.Or(a, c))

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=5
    )

    try:
        result = sample_models_from_formula(formula, Logic.QF_BOOL, options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")
    except Exception as e:
        print(f"Error: {e}")


def bitvector_example():
    """Example of sampling from a bit-vector formula."""
    print("\n=== Bit-Vector Sampling Example ===")

    # Create a bit-vector formula
    x = z3.BitVec('x', 8)
    y = z3.BitVec('y', 8)
    formula = z3.And(x + y > 10, x * 2 < 100)

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=5
    )

    try:
        result = sample_models_from_formula(formula, Logic.QF_BV, options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")
    except Exception as e:
        print(f"Error: {e}")


def lra_example():
    """Example of sampling from a linear real arithmetic formula."""
    print("\n=== Linear Real Arithmetic Sampling Example ===")

    # Create a linear real arithmetic formula
    x, y = z3.Reals('x y')
    formula = z3.And(x + y > 0, x - y < 1, x > -5, y < 5)

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=5
    )

    try:
        result = sample_models_from_formula(formula, Logic.QF_LRA, options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")
    except Exception as e:
        print(f"Error: {e}")


def lia_example():
    """Example of sampling from a linear integer arithmetic formula."""
    print("\n=== Linear Integer Arithmetic Sampling Example ===")

    # Create a linear integer arithmetic formula
    x, y = z3.Ints('x y')
    formula = z3.And(x + y > 0, x - y < 10, x > -5, y < 5)

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=5
    )

    try:
        result = sample_models_from_formula(formula, Logic.QF_LIA, options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")
    except Exception as e:
        print(f"Error: {e}")


def lira_example():
    """Example of sampling from a mixed linear integer and real arithmetic formula."""
    print("\n=== Mixed Linear Integer and Real Arithmetic Sampling Example ===")

    # Create a mixed linear integer and real arithmetic formula
    x = z3.Int('x')
    y = z3.Real('y')
    formula = z3.And(x + y > 0, x - y < 10, x > -5, y < 5)

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=5
    )

    try:
        result = sample_models_from_formula(formula, Logic.QF_LIRA, options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")
    except Exception as e:
        print(f"Error: {e}")


def advanced_example():
    """Example of more advanced sampling options."""
    print("\n=== Advanced Sampling Example ===")

    # Create a formula
    x, y = z3.Reals('x y')
    formula = z3.And(x + y > 0, x - y < 1, x > -5, y < 5)

    # Create a sampler directly
    try:
        sampler = create_sampler(Logic.QF_LRA)
        sampler.init_from_formula(formula)

        # Sample with custom options
        options = SamplingOptions(
            method=SamplingMethod.ENUMERATION,
            num_samples=3,
            timeout=10.0,
            random_seed=42,
            # Additional method-specific options
            use_blocking_clauses=True
        )

        result = sampler.sample(options)

        # Print the models
        print(f"Generated {len(result)} models:")
        for i, model in enumerate(result):
            print(f"Model {i + 1}: {model}")

        # Print statistics
        if result.stats:
            print("\nStatistics:")
            for key, value in result.stats.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")


def run_all_examples():
    """Run all examples."""
    boolean_example()
    bitvector_example()
    lra_example()
    lia_example()
    lira_example()
    advanced_example()


if __name__ == "__main__":
    run_all_examples()
