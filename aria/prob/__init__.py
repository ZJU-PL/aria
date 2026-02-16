"""
Probabilistic reasoning and weighted model counting/integration.

This package provides tools for probabilistic inference:

- `wmc/`: Weighted Model Counting (WMC) and Weighted Model Integration (WMI)
  
  WMC computes the weighted sum of satisfying assignments for propositional
  formulas. WMI extends this to continuous domains with density functions.

Key features:
- Weighted Model Counting over CNF formulas (DNNF and enumeration backends)
- Weighted Model Integration over continuous LRA/LIA formulas
- Multiple integration methods: sampling-based and region-based
- Support for common probability distributions: Uniform, Gaussian, Exponential, Beta

Example:
    >>> # Weighted Model Counting
    >>> from pysat.formula import CNF
    >>> from aria.prob.wmc import wmc_count, WMCBackend, WMCOptions
    >>> cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
    >>> weights = {1: 0.6, -1: 0.4, 2: 0.7, -2: 0.3, 3: 0.5, -3: 0.5}
    >>> result = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))
    
    >>> # Weighted Model Integration  
    >>> import z3
    >>> from aria.prob.wmc import wmi_integrate, WMIOptions, UniformDensity
    >>> x, y = z3.Reals('x y')
    >>> formula = z3.And(x + y > 0, x < 1, y < 1, x > 0, y > 0)
    >>> density = UniformDensity({'x': (0, 1), 'y': (0, 1)})
    >>> result = wmi_integrate(formula, density, WMIOptions(num_samples=10000))

For detailed API documentation, see `wmc/__init__.py`.
"""

from . import wmc

__all__ = ["wmc"]
