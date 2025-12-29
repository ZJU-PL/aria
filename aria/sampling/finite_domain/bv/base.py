"""
BitVector sampler implementation.

This module provides a sampler for bit-vector formulas.
"""

import random
from typing import Set

import z3

from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.utils.z3_expr_utils import get_variables, is_bv_sort


class BitVectorSampler(Sampler):
    """
    Sampler for bit-vector formulas.

    This class implements a sampler for bit-vector formulas using Z3.
    """

    def __init__(self, **_kwargs):
        """Initialize the bit-vector sampler."""
        self.formula = None
        self.variables = []

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

        # Extract variables from the formula
        self.variables = []
        for var in get_variables(formula):
            if is_bv_sort(var):
                self.variables.append(var)

        # Sort variables by name for deterministic ordering
        self.variables.sort(key=str)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """
        Generate samples according to the given options.

        Args:
            options: The sampling options

        Returns:
            A SamplingResult containing the generated samples
        """
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        # Set random seed if provided
        if options.random_seed is not None:
            random.seed(options.random_seed)

        # Create a solver with specific random seed
        solver = z3.Solver()
        if options.random_seed is not None:
            solver.set('random_seed', options.random_seed)
            solver.set('seed', options.random_seed)
        solver.add(self.formula)

        # Generate samples
        samples = []
        stats = {"time_ms": 0, "iterations": 0}

        for _ in range(options.num_samples):
            if solver.check() == z3.sat:
                model = solver.model()

                # Convert model to a dictionary
                sample = {}
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    sample[str(var)] = value.as_long()

                samples.append(sample)

                # Add blocking clause to prevent the same model
                block = []
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    block.append(var != value)

                solver.add(z3.Or(block))
                stats["iterations"] += 1
            else:
                break

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.

        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.

        Returns:
            A set of supported logics
        """
        return {Logic.QF_BV}
