"""
Simplified implementation of the Dissolve algorithm (Henry et al., 2015).

This package provides a practical, dependency-light version of the algorithm
that runs on top of PySAT. It implements:

- A scheduler that issues asynchronous Dilemma-rule splits (Algorithm 3)
- A round-based producer which selects split variables and merges results
- A light-weight UBTree-like structure for storing and prioritizing learnt
  clauses returned by workers
- Worker-side solving with assumptions, conflict budgets, and phases (when
  supported by the underlying SAT solver)

The focus is to make the code runnable and easy to experiment with in Python,
not to exactly reproduce every engineering detail of the original system.

Key APIs
--------
- Dissolve.solve(cnf, k=5, budget_conflicts=100000, max_rounds=None,
                 num_workers=None, solver_name="cd") -> (result, model)

Where `result` is one of aria.utils.types.SolverResult and `model` is an
optional assignment when satisfiable.
"""

from .dissolve import Dissolve, DissolveConfig, DissolveResult

__all__ = [
    "Dissolve",
    "DissolveConfig",
    "DissolveResult",
]
