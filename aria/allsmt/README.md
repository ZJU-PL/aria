# AllSMT - All Satisfying Models for SMT Formulas

This module provides tools for enumerating all satisfying models (AllSMT) for SMT formulas using various SMT solvers.

## Overview

The AllSMT module offers a unified interface for enumerating all satisfying models of SMT formulas using different
underlying solvers. It provides:

- A common interface for different solver backends
- Support for Z3, PySMT, and MathSAT
- Easy-to-use factory pattern for creating solvers
- Consistent API across all solver implementations
- **Uniform Z3 expression input support** for all solvers
- **Model limit control** to handle potentially infinite model sets

## Available Solvers

- **Z3**: Native Z3-based AllSMT solver using blocking clauses
- **PySMT**: PySMT-based AllSMT solver supporting multiple backends and Z3 expressions as input
- **MathSAT**: MathSAT-based AllSMT solver using native AllSAT capabilities

## Usage

### Basic Usage

```python
from aria.allsmt import create_allsmt_solver
from z3 import Ints, And

# Create a solver (default is Z3)
solver = create_allsmt_solver()

# Define variables and constraints
x, y = Ints('x y')
expr = And(x + y == 5, x > 0, y > 0)

# Solve the formula with a model limit
models = solver.solve(expr, [x, y], model_limit=10)

# Print the models
solver.print_models(verbose=True)

# Get the number of models
count = solver.get_model_count()

# Access the models directly
for model in solver.models:
    # Process model...
    pass
```

### Using Different Solvers

```python
# Create a Z3 solver
z3_solver = create_allsmt_solver("z3")

# Create a PySMT solver (now accepts Z3 expressions directly)
pysmt_solver = create_allsmt_solver("pysmt")

# Create a MathSAT solver
mathsat_solver = create_allsmt_solver("mathsat")
```

### Using PySMT Solver with Z3 Expressions

The PySMT solver now accepts Z3 expressions directly, providing a uniform interface:

```python
from aria.allsmt import create_allsmt_solver
from z3 import Ints, Bools, And, Or

# Create a PySMT solver
solver = create_allsmt_solver("pysmt")

# Define Z3 variables
x, y = Ints('x y')
a, b = Bools('a b')

# Define Z3 constraints
expr = And(
    a == (x + y > 0),
    Or(a, b),
    x > 0,
    y > 0
)

# Solve the Z3 formula with PySMT (limit to 20 models)
solver.solve(expr, [a, b, x, y], model_limit=20)
solver.print_models(verbose=True)
```

### Handling Potentially Infinite Model Sets

For integer or real formulas, the number of models can be infinite. The `model_limit` parameter prevents the solver from
running indefinitely:

```python
from aria.allsmt import create_allsmt_solver
from z3 import Reals, And

# Create a solver
solver = create_allsmt_solver()

# Define a formula with uncountably infinite models
a, b = Reals('a b')
expr = And(a > 0, b > 0, a + b == 1)

# Limit to 5 models
solver.solve(expr, [a, b], model_limit=5)
solver.print_models(verbose=True)
```

### Advanced Usage

See the `examples.py` file for more detailed examples of using the AllSMT API with different solvers and formulas.

## Architecture

The AllSMT module is designed with extensibility in mind:

- `AllSMTSolver`: Abstract base class defining the interface for all solver implementations
- `Z3AllSMTSolver`, `PySMTAllSMTSolver`, `MathSATAllSMTSolver`: Concrete implementations for specific solvers
- `Z3ToPySMTConverter`: Utility for converting Z3 expressions to PySMT format
- `AllSMTSolverFactory`: Factory for creating solver instances
- `create_allsmt_solver()`: Convenience function for creating solver instances

## Requirements

- Z3 (for Z3-based solver and Z3 expression input)
- PySMT (for PySMT-based solver)
- MathSAT (for MathSAT-based solver)

Note that each solver implementation will only be available if the corresponding library is installed.

## License

This module is part of the aria package and is subject to the same license terms. 