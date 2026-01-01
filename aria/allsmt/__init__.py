"""
AllSMT - All Satisfying Models for SMT formulas

This package provides tools for enumerating all satisfying models (AllSMT) for
SMT formulas using various SMT solvers.

Available solvers:
- Z3: Native Z3-based AllSMT solver
- PySMT: PySMT-based AllSMT solver supporting multiple backends
- MathSAT: MathSAT-based AllSMT solver using native AllSAT capabilities

Key features:
- Uniform interface for all solvers
- All solvers accept Z3 expressions as input
- Consistent API across different backends
- Model limit control for potentially infinite model sets

Usage:
    from aria.allsmt import create_allsmt_solver
    from z3 import Ints, And

    # Create a solver (default is Z3)
    solver = create_allsmt_solver()

    # Or specify a specific solver
    solver = create_allsmt_solver("mathsat")

    # Solve a formula with a model limit
    x, y = Ints('x y')
    expr = And(x + y == 5, x > 0, y > 0)
    models = solver.solve(expr, [x, y], model_limit=10)

    # Print the models
    solver.print_models(verbose=True)

    # Get the number of models
    count = solver.get_model_count()

    # Access the models directly
    for model in solver.models:
        # Process model...
        pass

Note on infinite models:
    For integer or real formulas, the number of models can be infinite.
    The model_limit parameter (default: 100) prevents the solver from
    running indefinitely by limiting the number of models returned.
"""

# Import the base classes
from .base import AllSMTSolver

# Import the factory
from .factory import AllSMTSolverFactory, create_allsmt_solver, create_solver

# Try to import specific solvers
try:
    from .z3_solver import Z3AllSMTSolver
except ImportError:
    pass

try:
    from .pysmt_solver import PySMTAllSMTSolver, Z3ToPySMTConverter
except ImportError:
    pass

try:
    from .mathsat_solver import MathSATAllSMTSolver
except ImportError:
    pass

# Define what's available in the public API
__all__ = [
    "AllSMTSolver",
    "AllSMTSolverFactory",
    "create_allsmt_solver",
    "create_solver",  # For backward compatibility
    "Z3AllSMTSolver",
    "PySMTAllSMTSolver",
    "MathSATAllSMTSolver",
    "Z3ToPySMTConverter",
]
