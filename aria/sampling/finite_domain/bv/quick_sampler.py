#!/usr/bin/env python3
# coding: utf-8
"""
QuickSampler implementation for bit-vector formulas.

Approach taken from:
  Rafael Dutra, Kevin Laeufer, Jonathan Bachrach and Koushik Sen:
  Efficient Sampling of SAT Solutions for Testing, ICSE 2018.
  https://github.com/RafaelTupynamba/quicksampler/

Note: The generated samples are currently not checked for whether they satisfy the given constraints!
"""

import random
import z3
import itertools
from functools import reduce
from typing import List, Dict, Any, Set, Optional

from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.utils.z3_expr_utils import get_variables, is_bv_sort


# https://stackoverflow.com/questions/39299015/sum-of-all-the-bits-in-a-bit-vector-of-z3
def _bvcount(b: z3.ExprRef):
    """Count the number of set bits in a bit-vector."""
    n = b.size()
    bits = [z3.Extract(i, i, b) for i in range(n)]
    bvs = [z3.Concat(z3.BitVecVal(0, n - 1), b) for b in bits]
    nb = reduce(lambda a, b: a + b, bvs)
    return nb


MAX_LEVEL = 6


def _bvsampler(constraints, target):
    """
    Internal implementation: Generate diverse samples using bit-flipping mutations.

    Args:
        constraints: The formula constraints
        target: The target bit-vector variable to sample (must be a single variable)

    Yields:
        Integer values for the target variable
    """
    n = target.size()

    solver = z3.Optimize()
    solver.add(constraints)
    delta = z3.BitVec('delta', n)
    result = z3.BitVec('result', n)
    solver.add(result == target)
    solver.minimize(_bvcount(delta))

    results = set()

    while True:
        # print('---------------------------')
        guess = z3.BitVecVal(random.getrandbits(n), n)

        solver.push()
        solver.add(result ^ delta == guess)

        if solver.check() != z3.sat:
            break

        model = solver.model()
        result0 = model[result].as_long()
        solver.pop()

        results.add(result0)
        yield result0

        # print('solver: ' + str(solver))
        # print('guess: ' + str(guess))
        # print('model: ' + str(model))

        mutations = {}

        solver.push()

        for i in range(n):
            # print('mutating bit ' + str(i))
            solver.push()
            goal = z3.BitVecVal(result0, n)
            solver.add(result ^ delta == goal)
            solver.add(z3.Extract(i, i, delta) == 0x1)

            if solver.check() == z3.sat:
                model = solver.model()
                result1 = model[result].as_long()

                if result1 not in results:
                    results.add(result1)
                    yield result1

                new_mutations = {}
                new_mutations[result1] = 1

                for value in mutations:
                    level = mutations[value]
                    if level > MAX_LEVEL:
                        continue

                    candidate = (result0 ^ ((result0 ^ value) | (result0 ^ result1)))
                    # print('yielding candidate ' + str(candidate) + ' at level ' + str(level))
                    if candidate not in results:
                        results.add(candidate)
                        yield candidate

                    new_mutations[candidate] = level + 1

                mutations.update(new_mutations)

            solver.pop()

        solver.pop()


class QuickBVSampler(Sampler):
    """
    QuickSampler for bit-vector formulas.

    This sampler uses optimization-guided sampling with bit-flipping mutations
    to generate diverse samples efficiently. It's particularly useful for testing
    and fuzzing applications.

    Note: Currently samples one target variable at a time. For formulas with multiple
    variables, it will sample the first bit-vector variable found.

    Reference:
        Rafael Dutra et al., "Efficient Sampling of SAT Solutions for Testing", ICSE 2018
        https://github.com/RafaelTupynamba/quicksampler/
    """

    def __init__(self, target_var: Optional[z3.ExprRef] = None, **kwargs: Any) -> None:
        """
        Initialize the QuickSampler.

        Args:
            target_var: Optional target variable to sample. If not provided,
                       will use the first bit-vector variable found in the formula.
        """
        self.formula: Optional[z3.ExprRef] = None
        self.target_var: Optional[z3.ExprRef] = target_var
        self.all_variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.

        Args:
            logic: The logic to check

        Returns:
            True if the sampler supports the logic, False otherwise
        """
        return logic == Logic.QF_BV

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """
        Initialize the sampler with a formula.

        Args:
            formula: The Z3 formula to sample from
        """
        self.formula = formula

        # Extract bit-vector variables
        self.all_variables = []
        for var in get_variables(formula):
            if is_bv_sort(var):
                self.all_variables.append(var)

        # Set target variable if not already set
        if self.target_var is None:
            if not self.all_variables:
                raise ValueError("No bit-vector variables found in formula")
            self.target_var = self.all_variables[0]

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        Generate samples using QuickSampler algorithm.

        Args:
            options: The sampling options

        Returns:
            A SamplingResult containing the generated samples
        """
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        if self.target_var is None:
            raise ValueError("No target variable specified")

        # Set random seed if provided
        if options.random_seed is not None:
            random.seed(options.random_seed)

        # Generate samples using the quicksampler algorithm
        samples: List[Dict[str, Any]] = []
        generator = _bvsampler(self.formula, self.target_var)

        for i, value in enumerate(generator):
            if i >= options.num_samples:
                break

            sample: Dict[str, Any] = {str(self.target_var): value}
            samples.append(sample)

        stats = {
            "time_ms": 0,
            "iterations": len(samples),
            "method": "quicksampler",
            "target_variable": str(self.target_var)
        }

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.

        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.ENUMERATION}  # Closest match for now

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.

        Returns:
            A set of supported logics
        """
        return {Logic.QF_BV}


def test_sampler():
    """Test the QuickSampler."""
    x = z3.BitVec('x', 16)
    y = z3.BitVec('y', 16)
    formula = z3.And(x > 1000, y < 10000, z3.Or(x < 4000, x > 5000))

    sampler = QuickBVSampler(target_var=x)
    sampler.init_from_formula(formula)
    result = sampler.sample(SamplingOptions(num_samples=10))

    print(f"Generated {len(result)} samples:")
    for i, sample in enumerate(result):
        print(f"  Sample {i+1}: {sample}")


if __name__ == '__main__':
    test_sampler()
