# coding: utf-8
"""
Implementation of Monadic Decomposition algorithm from CAV'14.

This module implements the monadic decomposition algorithm for decomposing
formulas into simpler components. The main function is mondec() which 
performs the decomposition.
TODO: to check
"""

from typing import List, Callable, Optional, Any
from z3 import *


def nu_ab(R: Callable, x: List, y: List, a: List, b: List) -> BoolRef:
    """Compute nu_ab predicate for monadic decomposition.
    
    Args:
        R: The relation to decompose
        x, y: Variables to decompose
        a, b: Witness points
    Returns:
        Z3 formula representing nu_ab
    """
    x_ = [Const(f"x_{i}", x[i].sort()) for i in range(len(x))]
    y_ = [Const(f"y_{i}", y[i].sort()) for i in range(len(y))]
    return Or(Exists(y_, R(x + y_) != R(a + y_)),
              Exists(x_, R(x_ + y) != R(x_ + b)))


def is_unsat(fml: BoolRef) -> bool:
    """Check if formula is unsatisfiable.
    
    Args:
        fml: Z3 formula to check
    Returns:
        True if formula is unsatisfiable
    """
    s = Solver()
    s.add(fml)
    return unsat == s.check()


def last_sat(solver: Solver,
             model: Optional[ModelRef],
             formulas: List[BoolRef]) -> Optional[ModelRef]:
    """Find the last satisfiable model in a sequence of formulas.
    
    Args:
        solver: Z3 solver instance
        model: Current model (can be None)
        formulas: List of formulas to check
    Returns:
        Last satisfiable model or None
    """
    if not formulas:
        return model

    solver.push()
    solver.add(formulas[0])

    if solver.check() == sat:
        model = last_sat(solver, solver.model(), formulas[1:])

    solver.pop()
    return model


def mondec(R: Callable, variables: List) -> BoolRef:
    """Perform monadic decomposition of relation R over variables.
    
    Args:
        R: Relation to decompose
        variables: List of variables to decompose
    Returns:
        Decomposed formula
    """
    phi = R(variables)
    if len(variables) == 1:
        return phi

    m = len(variables) // 2  # Use integer division
    x, y = variables[0:m], variables[m:]

    def decompose(nu: Solver, pi: BoolRef) -> BoolRef:
        # Base cases
        if is_unsat(And(pi, phi)):
            return BoolVal(False)
        if is_unsat(And(pi, Not(phi))):
            return BoolVal(True)

        # Build formula list based on heuristic
        formulas = [BoolVal(True)]
        if USE_HEURISTIC:
            formulas = [BoolVal(True), phi, pi]

        # Get satisfying model
        model = last_sat(nu, None, formulas)
        if model is None:
            raise RuntimeError("Failed to find consistent model")

        # Extract witness points
        a = [model.evaluate(z, True) for z in x]
        b = [model.evaluate(z, True) for z in y]

        # Recursive decomposition
        psi_ab = And(R(a + y), R(x + b))
        phi_a = mondec(lambda z: R(a + z), y)
        phi_b = mondec(lambda z: R(z + b), x)

        # Handle remaining cases
        nu.push()
        nu.add(nu_ab(R, x, y, a, b))
        t = decompose(nu, And(pi, psi_ab))
        f = decompose(nu, And(pi, Not(psi_ab)))
        nu.pop()

        return If(And(phi_a, phi_b), t, f)

    return decompose(Solver(), BoolVal(True))


def test_mondec(k: int) -> None:
    """Test monadic decomposition with example from Figure 3.
    
    Args:
        k: Bit-vector size parameter
    """
    try:
        R = lambda v: And(v[0] + v[1] == 2, v[0] < 1)
        bvs = BitVecSort(2 * k)
        x, y = Const("x", bvs), Const("y", bvs)

        res = mondec(R, [x, y])

        # Verify correctness
        if not is_unsat(res != R([x, y])):
            raise ValueError("Decomposition verification failed")

        print(f"mondec1({R([x, y])}) =")
        print(res)
        print(simplify(res))

    except Exception as e:
        print(f"Error in test_mondec: {str(e)}")


# Configuration
USE_HEURISTIC = True

if __name__ == "__main__":
    test_mondec(3)
