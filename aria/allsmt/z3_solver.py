"""
Z3-based AllSMT solver implementation.

This module provides an implementation of the AllSMT solver using Z3.
"""

from typing import List
from z3 import Solver, sat, Or, ModelRef, ExprRef

from .base import AllSMTSolver


class Z3AllSMTSolver(AllSMTSolver[ModelRef]):
    """
    Z3-based AllSMT solver implementation.

    This class implements the AllSMT solver interface using Z3 as the underlying solver.
    """

    def solve(self, expr: ExprRef, keys: List[ExprRef], model_limit: int = 100) -> List[ModelRef]:
        """
        Enumerate all satisfying models for the given expression over the specified keys.

        Args:
            expr: The Z3 expression/formula to solve
            keys: The Z3 variables to track in the models
            model_limit: Maximum number of models to generate (default: 100)

        Returns:
            List of Z3 models satisfying the expression
        """
        solver = Solver()
        solver.add(expr)
        self._reset_model_storage()

        while solver.check() == sat:
            model = solver.model()
            if self._add_model(model, model_limit):
                break

            # Create blocking clause to exclude the current model
            block = []
            for k in keys:
                block.append(k != model[k])
            solver.add(Or(block))

        return self._models

    def _format_model_verbose(self, model: ModelRef) -> None:
        """
        Print detailed information about a single Z3 model.

        Args:
            model: The Z3 model to print
        """
        for decl in model.decls():
            print(f"  {decl.name()} = {model[decl]}")


def demo() -> None:
    """Demonstrate the usage of the Z3-based AllSMT solver."""
    from z3 import Ints, And  # pylint: disable=import-outside-toplevel

    x, y = Ints('x y')
    expr = And(x + y == 5, x > 0, y > 0)

    solver = Z3AllSMTSolver()
    solver.solve(expr, [x, y], model_limit=10)
    solver.print_models(verbose=True)


if __name__ == "__main__":
    demo()
