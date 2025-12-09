#!/usr/bin/env python3

import sys
import random
# Add the aria directory to the Python path
from z3 import (
    Solver, sat, BoolRef, IntNumRef, RatNumRef, AlgebraicNumRef,
    IntVal, RealVal, BoolVal, is_int, is_real, is_bool, ExprRef,
    is_const, is_true, is_false
)

from aria.sampling.general_sampler.mcmc_sampler import MCMCSampler
from aria.sampling.base import SamplingOptions, SamplingResult, SamplingMethod
from z3 import Reals, Ints, And, Solver, sat, RealVal, IntVal


def test_manual_sampling():
    """Manual test to demonstrate getting multiple samples"""
    print("\nTest 1: Manual sampling for a circle in first quadrant")
    x, y = Reals('x y')
    formula = And(x ** 2 + y ** 2 < 1, x > 0, y > 0)

    # Create the sampler
    sampler = MCMCSampler(formula)

    # Check formula
    solver = Solver()
    solver.add(formula)
    assert solver.check() == sat, "Formula is not satisfiable"

    # Directly generate multiple samples
    samples = []
    model = solver.model()

    # Start with one sample
    x_val = 0.5
    y_val = 0.5
    samples.append({"x": x_val, "y": y_val})

    # Generate more samples with manual perturbation
    for i in range(4):
        # Perturb the values manually
        x_val = random.uniform(0.1, 0.9)
        y_val = random.uniform(0.1, 0.9)

        # Ensure within bounds (we need x^2 + y^2 < 1)
        while x_val ** 2 + y_val ** 2 >= 1.0 or x_val <= 0 or y_val <= 0:
            x_val = random.uniform(0.1, 0.9)
            y_val = random.uniform(0.1, 0.9)

        samples.append({"x": x_val, "y": y_val})

    # Print results
    print(f"Manually generated {len(samples)} samples:")
    for i, sample in enumerate(samples):
        x_val = sample["x"]
        y_val = sample["y"]
        print(f"Sample {i + 1}: x = {x_val}, y = {y_val}")
        print(f"  Verification: x^2 + y^2 = {x_val ** 2 + y_val ** 2} < 1")


def test_mcmc_integer_range():
    """Test simple integer range with MCMC sampler"""
    print("\nTest 2: Simple integer range with MCMC")

    # Use integers for a simpler example
    a = Ints('a')[0]
    formula = And(a >= 1, a <= 10)

    sampler = MCMCSampler(formula, step_size=1.0)

    # Create sampling options with additional_options as kwargs
    options = SamplingOptions(
        method=SamplingMethod.MCMC,
        num_samples=5,
        burn_in=2,
        step_size=1.0
    )

    result = sampler.sample(options)

    # Print results
    print(f"MCMC generated {len(result)} samples:")
    for i, sample in enumerate(result):
        a_val = sample['a'].as_long()
        print(f"Sample {i + 1}: a = {a_val}")

    print(f"Sampling stats: {result.stats}")


def test_complex_integer_equation():
    """Test a more complex integer equation with MCMC sampler"""
    print("\nTest 3: Complex integer equation")

    # Integer equation with multiple variables
    a, b, c = Ints('a b c')
    formula = And(a > 0, b > 0, c > 0, a + b + c == 10, a <= b, b <= c)

    sampler = MCMCSampler(formula, step_size=1.0)

    # Create sampling options with additional_options as kwargs
    options = SamplingOptions(
        method=SamplingMethod.MCMC,
        num_samples=5,
        burn_in=5,
        step_size=1.0
    )

    result = sampler.sample(options)

    # Print results
    print(f"MCMC generated {len(result)} samples:")
    for i, sample in enumerate(result):
        a_val = sample['a'].as_long()
        b_val = sample['b'].as_long()
        c_val = sample['c'].as_long()
        print(f"Sample {i + 1}: a = {a_val}, b = {b_val}, c = {c_val}")
        print(
            f"  Verification: a + b + c = {a_val + b_val + c_val} == 10, a <= b <= c: {a_val <= b_val and b_val <= c_val}")

    print(f"Sampling stats: {result.stats}")


if __name__ == "__main__":
    test_manual_sampling()
    test_mcmc_integer_range()
    test_complex_integer_equation()
