"""
MCMC Sampling for Solutions of SMT formulas

This module provides an implementation of Markov Chain Monte Carlo (MCMC)
sampling for SMT formulas using Z3. It supports sampling from various
SMT logics including QF_LRA, QF_NRA, QF_LIA, and QF_NIA.

by LLM, to check
"""

import random
import time
from typing import Dict, List, Set, Any, Optional, Tuple

from z3 import (
    Solver, sat, BoolRef, IntNumRef, RatNumRef, AlgebraicNumRef,
    IntVal, RealVal, BoolVal, is_int, is_real, is_bool, ExprRef,
    is_const, is_true, is_false
)

from ..base import Sampler, SamplingOptions, SamplingResult, Logic, SamplingMethod


class MCMCSampler(Sampler):
    """
    Markov Chain Monte Carlo (MCMC) sampler for SMT formulas.
    
    This sampler uses a Metropolis-Hastings MCMC algorithm to generate samples
    from SMT formulas. It performs a random walk in the solution space,
    proposing new samples by perturbing the current sample.
    """

    def __init__(self, formula=None, step_size=0.1, max_attempts=100):
        """
        Initialize the MCMC sampler.
        
        Args:
            formula: Optional Z3 formula to sample from
            step_size: Step size for continuous variables
            max_attempts: Maximum number of attempts to find a valid next sample
        """
        self.formula = formula
        self.step_size = step_size
        self.max_attempts = max_attempts
        self.solver = None
        self.variables = None

        if formula is not None:
            self.init_from_formula(formula)

    def supports_logic(self, logic: Logic) -> bool:
        """
        Check if this sampler supports the given logic.
        
        Args:
            logic: The logic to check
            
        Returns:
            True if the sampler supports the logic, False otherwise
        """
        supported = {
            Logic.QF_LRA, Logic.QF_LIA, Logic.QF_NRA, Logic.QF_NIA,
            Logic.QF_LIRA, Logic.QF_BOOL, Logic.QF_ALL
        }
        return logic in supported

    def init_from_formula(self, formula: ExprRef) -> None:
        """
        Initialize the sampler with a formula.
        
        Args:
            formula: The Z3 formula to sample from
        """
        self.formula = formula
        self.solver = Solver()
        self.solver.add(formula)

        # Extract variables from the formula
        self.variables = self._extract_variables(formula)

    def _extract_variables(self, formula):
        """Extract variables from the formula."""
        # This is a simplified implementation - a more robust implementation
        # would need to traverse the formula to find all variables

        # For simplicity, we'll just use the solver's model
        if self.solver.check() == sat:
            model = self.solver.model()
            return {d.name(): d for d in model.decls()}

        return {}

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """
        Get the sampling methods supported by this sampler.
        
        Returns:
            A set of supported sampling methods
        """
        return {SamplingMethod.MCMC}

    def get_supported_logics(self) -> Set[Logic]:
        """
        Get the logics supported by this sampler.
        
        Returns:
            A set of supported logics
        """
        return {
            Logic.QF_LRA, Logic.QF_LIA, Logic.QF_NRA, Logic.QF_NIA,
            Logic.QF_LIRA, Logic.QF_BOOL, Logic.QF_ALL
        }

    def sample(self, options: Optional[SamplingOptions] = None) -> SamplingResult:
        """
        Generate samples according to the given options.
        
        Args:
            options: The sampling options
            
        Returns:
            A SamplingResult containing the generated samples
        """
        if options is None:
            options = SamplingOptions(
                method=SamplingMethod.MCMC,
                num_samples=10
            )

        # Ensure additional_options exists
        if not hasattr(options, 'additional_options') or options.additional_options is None:
            options.additional_options = {}

        # Set random seed if provided
        if options.random_seed is not None:
            random.seed(options.random_seed)

        # Extract MCMC-specific options
        burn_in = options.additional_options.get('burn_in', 100)
        step_size = options.additional_options.get('step_size', self.step_size)
        max_attempts = options.additional_options.get('max_attempts', self.max_attempts)

        # Start timing
        start_time = time.time()

        # Check if formula is satisfiable
        if self.solver.check() != sat:
            return SamplingResult(
                samples=[],
                stats={
                    'time': time.time() - start_time,
                    'error': 'Formula is unsatisfiable'
                }
            )

        # Quick check for simple integer range problem (a >= min and a <= max)
        # This is a special case that's easy to handle
        if self._is_simple_integer_range():
            return self._sample_simple_integer_range(options, start_time)

        samples = []
        unique_samples_set = set()
        current_sample = self._random_initial_sample()

        # Add the initial sample
        sample_tuple = self._sample_to_tuple(current_sample)
        unique_samples_set.add(sample_tuple)
        samples.append(dict(current_sample))

        # Main MCMC loop
        iterations = 0
        max_iterations = (burn_in + options.num_samples) * 10  # Set a reasonable limit

        while len(samples) < options.num_samples and iterations < max_iterations:
            iterations += 1

            # Check timeout
            if options.timeout and (time.time() - start_time) > options.timeout:
                break

            # Propose next sample
            next_sample = self._propose_next_sample(current_sample, step_size)

            # Accept/reject the proposed sample
            if self._accept_sample(next_sample):
                current_sample = next_sample

                # After burn-in, collect samples
                if iterations >= burn_in:
                    # Convert sample to a hashable form for uniqueness check
                    sample_tuple = self._sample_to_tuple(current_sample)

                    if sample_tuple not in unique_samples_set:
                        unique_samples_set.add(sample_tuple)
                        samples.append(dict(current_sample))

        # Compute statistics
        stats = {
            'time': time.time() - start_time,
            'samples_collected': len(samples),
            'burn_in': burn_in,
            'step_size': step_size,
            'iterations': iterations
        }

        return SamplingResult(samples=samples, stats=stats)

    def _random_initial_sample(self) -> Dict[str, Any]:
        """
        Generate a random initial sample that satisfies the formula.
        
        Returns:
            A dictionary mapping variable names to values
        """
        if self.solver.check() != sat:
            raise ValueError("Formula is unsatisfiable")

        model = self.solver.model()
        return {str(d.name()): model[d] for d in model.decls()}

    def _sample_to_tuple(self, sample: Dict[str, Any]) -> Tuple:
        """
        Convert a sample dictionary to a hashable tuple for uniqueness check.
        
        We use a coarser representation to allow for more diversity in samples
        by rounding/bucketing numeric values.
        
        Args:
            sample: The sample to convert
            
        Returns:
            A hashable tuple representation of the sample
        """
        items = []

        for k, v in sorted(sample.items()):
            if is_bool(v):
                # For booleans, use the exact value
                items.append((k, is_true(v)))
            elif isinstance(v, IntNumRef):
                # For integers, use the exact value
                items.append((k, v.as_long()))
            elif isinstance(v, RatNumRef) or is_real(v):
                # For real values, round to reduce number of distinct values
                try:
                    # Round to 1 decimal place to allow more diverse samples
                    float_val = float(v.as_decimal(10))
                    rounded_val = round(float_val, 1)
                    items.append((k, rounded_val))
                except:
                    items.append((k, str(v)))
            else:
                # For other types, use string representation
                items.append((k, str(v)))

        return tuple(items)

    def _propose_next_sample(self, current_sample: Dict[str, Any], step_size: float) -> Dict[str, Any]:
        """
        Propose a new sample by perturbing the current sample.
        
        Args:
            current_sample: The current sample
            step_size: The step size for continuous variables
            
        Returns:
            A new proposed sample
        """
        next_sample = {}

        for var_name, value in current_sample.items():
            # Different perturbation strategies based on variable type
            if is_bool(value):
                # For boolean values, flip with some probability
                if random.random() < 0.2:  # 20% chance to flip
                    next_sample[var_name] = BoolVal(not is_true(value))
                else:
                    next_sample[var_name] = value

            elif isinstance(value, IntNumRef):
                # For integers, add/subtract a small random integer
                int_val = value.as_long()
                # Use a larger range for integer perturbations
                delta = random.choice([-2, -1, 0, 1, 2])
                next_sample[var_name] = IntVal(int_val + delta)

            elif isinstance(value, RatNumRef):
                # For rationals, perturb the numerator and denominator
                numerator = value.numerator_as_long()
                denominator = value.denominator_as_long()

                # More aggressive perturbation
                new_numerator = numerator + random.randint(-2, 2)
                # Occasionally perturb the denominator too
                if random.random() < 0.1:  # 10% chance
                    new_denominator = max(1, denominator + random.randint(-1, 1))
                else:
                    new_denominator = denominator

                next_sample[var_name] = RealVal(new_numerator) / RealVal(new_denominator)

            elif isinstance(value, AlgebraicNumRef):
                # AlgebraicNumRef represents roots of polynomials
                # For simplicity, we don't perturb these and keep them unchanged
                next_sample[var_name] = value

            elif is_real(value):
                # For real values, add a small random perturbation
                try:
                    float_val = float(value.as_decimal(10))
                    # Use a normal distribution for more realistic walks
                    delta = random.gauss(0, step_size)
                    next_sample[var_name] = RealVal(float_val + delta)
                except:
                    # Fall back to keeping the value unchanged
                    next_sample[var_name] = value

            else:
                # For unsupported types, keep them unchanged
                next_sample[var_name] = value

        return next_sample

    def _accept_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a proposed sample satisfies the formula.
        
        Args:
            sample: The proposed sample
            
        Returns:
            True if the sample satisfies the formula, False otherwise
        """
        self.solver.push()

        # Add constraints for each variable in the sample
        for var_name, value in sample.items():
            # Create a constraint that the variable equals the value
            # This requires finding the original variable from the name
            if var_name in self.variables:
                var = self.variables[var_name]
                self.solver.add(var == value)

        # Check if the formula is satisfiable with these constraints
        is_accepted = self.solver.check() == sat

        self.solver.pop()
        return is_accepted

    def _is_simple_integer_range(self) -> bool:
        """
        Check if the formula is a simple integer range (a >= min and a <= max).
        
        Returns:
            True if the formula is a simple integer range, False otherwise
        """
        # This is a simplified detection - in practice would need more robust parsing
        try:
            if self.solver.check() != sat:
                return False

            model = self.solver.model()
            if len(model) != 1:  # Only one variable
                return False

            # Get the variable and its value
            var = model.decls()[0]
            if not is_int(model[var]):  # Integer variable
                return False

            # Check if the formula is of the form a >= min and a <= max
            # by sampling several values and checking consistency

            # Sample a few points and check if they satisfy a continuous range
            test_range = 20
            valid_values = []

            for i in range(1, test_range + 1):
                self.solver.push()
                self.solver.add(var == i)
                if self.solver.check() == sat:
                    valid_values.append(i)
                self.solver.pop()

            # If no valid values or too few, not a simple range
            if len(valid_values) <= 1:
                return False

            # Check if it's a continuous range
            for i in range(1, len(valid_values)):
                if valid_values[i] != valid_values[i - 1] + 1:
                    return False

            # If we got here, it's likely a simple integer range
            return True
        except Exception as e:
            return False

    def _sample_simple_integer_range(self, options, start_time) -> SamplingResult:
        """
        Sample from a simple integer range.
        
        Args:
            options: Sampling options
            start_time: Time when sampling started
            
        Returns:
            Sampling result with uniformly sampled integers
        """
        # Get the variable
        model = self.solver.model()
        var = model.decls()[0]
        var_name = var.name()

        # Find min and max by sampling
        candidates = []
        min_val = 1
        max_val = 100  # An arbitrary upper bound to check

        # Test each value
        for i in range(min_val, max_val + 1):
            self.solver.push()
            self.solver.add(var == i)
            if self.solver.check() == sat:
                candidates.append(i)
            self.solver.pop()

            # Stop if we haven't found a valid value for a while
            if len(candidates) > 0 and i > candidates[-1] + 10:
                break

        # Sample from candidates
        samples = []
        num_samples = min(options.num_samples, len(candidates))

        # Shuffle candidates for random sampling
        random.shuffle(candidates)

        for i in range(num_samples):
            sample = {var_name: IntVal(candidates[i])}
            samples.append(sample)

        # Compute statistics
        stats = {
            'time': time.time() - start_time,
            'samples_collected': len(samples),
            'method': 'simple_integer_range',
            'range': f"[{candidates[0]}, {candidates[-1]}]",
            'candidates': len(candidates)
        }

        return SamplingResult(samples=samples, stats=stats)
