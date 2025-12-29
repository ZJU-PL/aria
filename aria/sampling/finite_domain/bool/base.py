"""
Boolean sampler implementation.

This module provides a sampler for Boolean formulas.

TODO:
   - Support using pysat solver for parallel sampling
   - Support pyunigen sampler for uniform sampling
"""

import random
from typing import Set, Dict, Any, List, Optional

import z3

from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult
from aria.utils.z3_expr_utils import get_variables, is_bool_sort


class BooleanSampler(Sampler):
    """
    Sampler for Boolean formulas.

    This class implements a sampler for Boolean formulas using Z3.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Boolean sampler."""
        self.formula: Optional[z3.ExprRef] = None
        self.variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.

        Args:
            logic: The logic to check

        Returns:
            True if the sampler supports the logic, False otherwise
        """
        return logic == Logic.QF_BOOL

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
            if is_bool_sort(var):
                self.variables.append(var)

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

        # Create a solver
        solver = z3.Solver()
        solver.add(self.formula)

        # Generate samples
        samples: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {"time_ms": 0, "iterations": 0}

        for _ in range(options.num_samples):
            if solver.check() == z3.sat:
                model = solver.model()

                # Convert model to a dictionary
                sample: Dict[str, Any] = {}
                for var in self.variables:
                    value = model.evaluate(var, model_completion=True)
                    sample[str(var)] = bool(value)

                samples.append(sample)

                # Add blocking clause to prevent the same model
                block: List[z3.ExprRef] = []
                for var in self.variables:
                    if z3.is_true(model[var]):
                        block.append(var == z3.BoolVal(False))
                    else:
                        block.append(var == z3.BoolVal(True))

                solver.add(z3.Or(block))
                stats["iterations"] += 1
            else:
                break

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Return the set of supported sampling methods.

        Returns:
            Set of supported sampling methods
        """
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Return the set of supported logics.

        Returns:
            Set of supported logics
        """
        return {Logic.QF_BOOL}
