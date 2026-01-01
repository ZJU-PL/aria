"""
Example usage of the Polyhorn API
"""

from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    And,
    Equals,
    ForAll,
    Implies,
    Minus,
    Real,
    Solver,
    Symbol,
)
from pysmt.typing import REAL

from aria.quant.polyhorn.main import execute

# Create symbols
x = Symbol("x", REAL)
y = Symbol("y", REAL)
z = Symbol("z", REAL)
l = Symbol("l", REAL)

# Create a solver
solver = Solver(name="z3")

# Add constraints to the solver
solver.add_assertion(z < y)  # (assert (< z y))
solver.add_assertion(
    ForAll(
        [l],  # (assert (forall ((l Real)) ...))
        Implies(  # (=> (and ...) (and ...))
            And(  # (and (= x l) (>= x 1) (<= x 3))
                Equals(x, l), GE(x, Real(1)), LE(x, Real(3))
            ),
            And(  # (and (<= x (- 10 z)) (= l 2) (= z (- x y)) (> z -1))
                LE(x, Minus(Real(10), z)),
                Equals(l, Real(2)),
                Equals(z, Minus(x, y)),
                GT(z, Real(-1)),
            ),
        ),
    )
)

# Create a configuration dictionary
config = {"theorem_name": "farkas", "solver_name": "z3"}

print(execute(solver, config))
