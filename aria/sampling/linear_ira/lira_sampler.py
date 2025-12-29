"""
Linear Integer and Real Arithmetic sampler implementation.

This module provides a sampler for linear integer and real arithmetic formulas.
"""

import random
from typing import Set

import z3

from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.utils.z3_expr_utils import get_variables, is_int_sort, is_real_sort


class LIRASampler(Sampler):
    """
    Sampler for linear integer and real arithmetic formulas.

    This class implements a sampler for linear integer and real arithmetic formulas using Z3.
    """

    def __init__(self, **_kwargs):
        """Initialize the LIRA sampler."""
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
        return logic in [Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA]

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
            if is_int_sort(var) or is_real_sort(var):
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
                    if is_int_sort(var):
                        sample[str(var)] = value.as_long()
                    else:  # Real
                        # Convert rational numbers to float
                        try:
                            # Try as_decimal first (works for simple rationals)
                            sample[str(var)] = float(value.as_decimal(10))
                        except (ValueError, TypeError, AttributeError):
                            try:
                                # Try direct conversion for rationals
                                if value.is_rational():
                                    # Use arithmetic to avoid direct comparison
                                    num = value.numerator()
                                    den = value.denominator()
                                    if den != 0:  # This should work since we check is_rational
                                        sample[str(var)] = float(num) / float(den)
                                    else:
                                        sample[str(var)] = float('inf')
                                else:
                                    # Fallback to string conversion
                                    sample[str(var)] = float(str(value))
                            except (ValueError, TypeError, AttributeError):
                                # Last resort - use 0.0
                                sample[str(var)] = 0.0

                samples.append(sample)

                # Add blocking clause to prevent the same model
                # For reals, we need to be careful about exact equality
                block = []
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    if is_int_sort(var):
                        block.append(var != value)
                    else:  # Real
                        # For reals, convert to float for comparison
                        try:
                            # Try as_decimal first
                            float_value = float(value.as_decimal(10))
                        except (ValueError, TypeError, AttributeError):
                            try:
                                # Try rational conversion
                                if value.is_rational():
                                    num = value.numerator()
                                    den = value.denominator()
                                    float_value = float(num) / float(den) if den != 0 else 0.0
                                else:
                                    float_value = 0.0
                            except (ValueError, TypeError, AttributeError):
                                float_value = 0.0

                        # For reals, we add a small delta to avoid numerical issues
                        delta = 0.001
                        block.append(z3.Or(var < float_value - delta, var > float_value + delta))

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
        return {SamplingMethod.ENUMERATION, SamplingMethod.DIKIN_WALK}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.

        Returns:
            A set of supported logics
        """
        return {Logic.QF_LRA, Logic.QF_LIA, Logic.QF_LIRA}


class LIASampler(Sampler):
    """Linear Integer Arithmetic sampler."""

    def __init__(self, **options):
        Sampler.__init__(self, **options)

        self.conjuntion_sampler = None
        self.number_samples = 0

    def sample(self, number=1):
        """
        External interface
        """
        self.number_samples = number
        return self.sample_via_enumeration()

    def sample_via_smt_enumeration(self):
        """
        Call an SMT solver iteratively (block sampled models)
        """
        raise NotImplementedError

    def sample_via_smt_random_seed(self):
        """
        Call an SMT solver iteratively (no blocking, but give the solver diferent
        random seeds)
        """
        raise NotImplementedError
