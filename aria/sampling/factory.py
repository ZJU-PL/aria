"""
Factory for creating samplers.

This module provides a factory for creating instances of different sampler implementations.
"""

from typing import Dict, Type, Optional, Any, Set, List
import z3

from .base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult


class SamplerFactory:
    """
    Factory for creating samplers.

    This class provides methods for registering and creating instances of different
    sampler implementations.
    """

    _samplers: Dict[Logic, List[Type[Sampler]]] = {}

    @classmethod
    def register(cls, logic: Logic, sampler_class: Type[Sampler]):
        """
        Register a sampler class with the factory.

        Args:
            logic: The logic to register the sampler for
            sampler_class: The sampler class to register
        """
        if logic not in cls._samplers:
            cls._samplers[logic] = []

        cls._samplers[logic].append(sampler_class)

    @classmethod
    def create(cls, logic: Logic, method: Optional[SamplingMethod] = None, **kwargs) -> Sampler:
        """
        Create an instance of a sampler for the specified logic and method.

        Args:
            logic: The logic to create a sampler for
            method: Optional sampling method to use
            **kwargs: Additional arguments to pass to the sampler constructor

        Returns:
            An instance of a sampler for the specified logic

        Raises:
            ValueError: If no sampler is available for the specified logic or method
        """
        if logic not in cls._samplers or not cls._samplers[logic]:
            available = ", ".join(str(l) for l in cls._samplers.keys())
            raise ValueError(f"No sampler available for logic {logic}. Available logics: {available}")

        # If method is specified, find a sampler that supports it
        if method:
            for sampler_class in cls._samplers[logic]:
                sampler = sampler_class(**kwargs)
                if method in sampler.get_supported_methods():
                    return sampler

            raise ValueError(f"No sampler available for logic {logic} and method {method}")

        # Otherwise, return the first registered sampler
        return cls._samplers[logic][0](**kwargs)

    @classmethod
    def available_logics(cls) -> Set[Logic]:
        """
        Get a set of available logics.

        Returns:
            Set of available logics
        """
        return set(cls._samplers.keys())

    @classmethod
    def available_methods(cls, logic: Logic) -> Set[SamplingMethod]:
        """
        Get a set of available methods for the specified logic.

        Args:
            logic: The logic to get available methods for

        Returns:
            Set of available methods
        """
        if logic not in cls._samplers:
            return set()

        methods = set()
        for sampler_class in cls._samplers[logic]:
            sampler = sampler_class()
            methods.update(sampler.get_supported_methods())

        return methods


# Try to import and register available samplers
try:
    from .finite_domain.bool_sampler import BooleanSampler

    SamplerFactory.register(Logic.QF_BOOL, BooleanSampler)
except ImportError:
    pass

try:
    from .finite_domain.bv_sampler import BitVectorSampler

    SamplerFactory.register(Logic.QF_BV, BitVectorSampler)
except ImportError:
    pass

try:
    from .linear_ira.lira_sampler import LIRASampler

    SamplerFactory.register(Logic.QF_LRA, LIRASampler)
    SamplerFactory.register(Logic.QF_LIA, LIRASampler)
    SamplerFactory.register(Logic.QF_LIRA, LIRASampler)
except ImportError:
    pass

# Register general-purpose MCMC sampler where applicable
try:
    from .general_sampler.mcmc_sampler import MCMCSampler

    for _logic in [
        Logic.QF_LRA,
        Logic.QF_LIA,
        Logic.QF_NRA,
        Logic.QF_NIA,
        Logic.QF_LIRA,
        Logic.QF_BOOL,
        Logic.QF_ALL,
    ]:
        SamplerFactory.register(_logic, MCMCSampler)
except ImportError:
    pass


def create_sampler(logic: Logic, method: Optional[SamplingMethod] = None, **kwargs) -> Sampler:
    """
    Convenience function to create a sampler instance.

    Args:
        logic: The logic to create a sampler for
        method: Optional sampling method to use
        **kwargs: Additional arguments to pass to the sampler constructor

    Returns:
        An instance of a sampler for the specified logic
    """
    return SamplerFactory.create(logic, method, **kwargs)


def sample_models_from_formula(formula: z3.ExprRef,
                               logic: Logic,
                               options: Optional[SamplingOptions] = None) -> SamplingResult:
    """
    High-level API for sampling models (solutions) from a formula.

    Args:
        formula: The Z3 formula to sample models from
        logic: The logic of the formula
        options: Optional sampling options

    Returns:
        A SamplingResult containing the generated models
    """
    if options is None:
        options = SamplingOptions()

    sampler = create_sampler(logic, options.method)
    sampler.init_from_formula(formula)
    return sampler.sample(options)


# For backward compatibility, but will be deprecated in future versions
def sample_formula(formula: z3.ExprRef,
                   logic: Logic,
                   options: Optional[SamplingOptions] = None) -> SamplingResult:
    """
    High-level API for sampling models from a formula.

    This function is deprecated and will be removed in a future version.
    Please use sample_models_from_formula() instead.

    Args:
        formula: The Z3 formula to sample models from
        logic: The logic of the formula
        options: Optional sampling options

    Returns:
        A SamplingResult containing the generated models
    """
    import warnings
    warnings.warn(
        "sample_formula() is deprecated and will be removed in a future version. "
        "Please use sample_models_from_formula() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return sample_models_from_formula(formula, logic, options)


def demo():
    """Demonstrate the usage of the sampler factory."""
    import z3

    # Create a simple formula
    x, y = z3.Reals("x y")
    formula = z3.And(x + y > 0, x - y < 1)

    # Sample from the formula
    result = sample_models_from_formula(formula, Logic.QF_LRA, SamplingOptions(num_samples=5))

    # Print the samples
    print(f"Generated {len(result)} models:")
    for i, sample in enumerate(result):
        print(f"Model {i + 1}: {sample}")


if __name__ == "__main__":
    demo()
