"""
Model Sampling

This package provides tools for sampling models from SMT formulas across various logics.

Available logics:
- QF_BOOL: Quantifier-free Boolean logic
- QF_BV: Quantifier-free bit-vector logic
- QF_LRA: Quantifier-free linear real arithmetic
- QF_LIA: Quantifier-free linear integer arithmetic
- QF_NRA: Quantifier-free non-linear real arithmetic
- QF_NIA: Quantifier-free non-linear integer arithmetic
- QF_LIRA: Quantifier-free linear integer and real arithmetic

Available sampling methods:
- ENUMERATION: Simple enumeration of models
- MCMC: Markov Chain Monte Carlo
- REGION: Region-based sampling
- SEARCH_TREE: Search tree-based sampling
- HASH_BASED: Hash-based sampling
- DIKIN_WALK: Dikin walk for continuous domains

Usage:
    from aria.sampling import sample_models_from_formula, Logic, SamplingOptions, SamplingMethod
    import z3

    # Create a formula
    x, y = z3.Reals('x y')
    formula = z3.And(x + y > 0, x - y < 1)

    # Sample models from the formula
    options = SamplingOptions(
        method=SamplingMethod.ENUMERATION,
        num_samples=10
    )
    result = sample_models_from_formula(formula, Logic.QF_LRA, options)

    # Print the models
    for i, model in enumerate(result):
        print(f"Model {i+1}: {model}")
"""

# Import base classes and enums
from .base import Logic, SamplingMethod, SamplingOptions, SamplingResult, Sampler

# Import factory functions
from .factory import (
    create_sampler,
    sample_models_from_formula,
    sample_formula,
    SamplerFactory,
)

# Define what's available in the public API
__all__ = [
    # Base classes and enums
    "Logic",
    "SamplingMethod",
    "SamplingOptions",
    "SamplingResult",
    "Sampler",
    # Factory functions
    "create_sampler",
    "sample_models_from_formula",
    "sample_formula",  # For backward compatibility
    "SamplerFactory",
]
