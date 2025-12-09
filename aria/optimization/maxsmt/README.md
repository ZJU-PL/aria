# MaxSMT

This module provides implementations of various MaxSMT solving algorithms.

## Module Structure

- `base.py`: Base class and common utilities for MaxSMT solvers
- `core_guided.py`: Core-guided approach (Fu-Malik / MSUS3 from SAT'13)
- `ihs.py`: Implicit Hitting Set approach (from IJCAR'18)
- `local_search.py`: Local search for MaxSMT(LIA) (from FM'24)
- `z3_optimize.py`: Implementation using Z3's built-in Optimize engine
- `__init__.py`: Main interface and examples

## Usage

```python
from aria.optimization.maxsmt import solve_maxsmt, MaxSMTSolver

# Using the convenience function
sat, model, cost = solve_maxsmt(hard_constraints, soft_constraints, weights, algorithm="core-guided")

# Or create a solver instance
solver = MaxSMTSolver(algorithm="ihs")
solver.add_hard_constraints(hard_constraints)
solver.add_soft_constraints(soft_constraints, weights)
sat, model, cost = solver.solve()
```

## Algorithms

- SAT'13: Modular approach to MaxSAT modulo theories. Cimatti, A., Griggio, A., Schaafsma, B.J., Sebastiani, R.
- IJCAR'18: Implicit hitting set algorithms for maximum satisfiability modulo theories. Fazekas, K., Bacchus, F., Biere, A.
- FM'24: A Local Search Algorithm for MaxSMT(LIA)

## Applications

- AAAI'19: Concurrency debugging with MaxSMT. Terra-Neves, M., Machado, N., Lynce, I., Manquinho, V.