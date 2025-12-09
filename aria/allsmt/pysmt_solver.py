"""
PySMT-based AllSMT solver implementation.

This module provides an implementation of the AllSMT solver using PySMT.
It accepts Z3 expressions as input and converts them to PySMT format internally.
"""

from typing import List, Any, Dict, Optional, Tuple, Union
import z3
from z3 import ExprRef
from pysmt.fnode import FNode
from pysmt.oracles import get_logic
from pysmt.shortcuts import Solver, Not, EqualsOrIff, And, Or

from aria.allsmt.base import AllSMTSolver

# Type alias for PySMT model (dictionary mapping variables to values)
PySMTModel = Dict[FNode, FNode]


class Z3ToPySMTConverter:
    """Handles conversion between Z3 and PySMT expressions."""

    @staticmethod
    def to_pysmt_vars(z3vars: List[ExprRef]) -> List[FNode]:
        """
        Convert Z3 variables to PySMT variables.

        Args:
            z3vars: List of Z3 expression references

        Returns:
            List of PySMT Symbol objects

        Raises:
            NotImplementedError: If unsupported Z3 type is encountered
        """
        from pysmt.shortcuts import Symbol
        from pysmt.typing import INT, REAL, BVType, BOOL

        type_mapping = {
            z3.is_int: INT,
            z3.is_real: REAL,
            z3.is_bool: BOOL
        }

        result = []
        for var in z3vars:
            var_name = var.decl().name()

            # Handle BV type separately due to size parameter
            if z3.is_bv(var):
                result.append(Symbol(var_name, BVType(var.sort().size())))
                continue

            # Handle other types
            for type_check, pysmt_type in type_mapping.items():
                if type_check(var):
                    result.append(Symbol(var_name, pysmt_type))
                    break
            else:
                raise NotImplementedError(f"Unsupported Z3 type for variable: {var}")

        return result

    @staticmethod
    def convert(z3_formula: ExprRef) -> Tuple[List[FNode], FNode]:
        """
        Convert Z3 formula to PySMT format.

        Args:
            z3_formula: Z3 expression to convert

        Returns:
            Tuple of (PySMT variables, PySMT formula)
        """
        from aria.utils.z3_expr_utils import get_variables

        z3_vars = get_variables(z3_formula)
        pysmt_vars = Z3ToPySMTConverter.to_pysmt_vars(z3_vars)

        # Convert formula using Z3 solver
        z3_solver = Solver(name='z3')
        pysmt_formula = z3_solver.converter.back(z3_formula)

        return pysmt_vars, pysmt_formula


class PySMTAllSMTSolver(AllSMTSolver[PySMTModel]):
    """
    PySMT-based AllSMT solver implementation.

    This class implements the AllSMT solver interface using PySMT as the underlying solver.
    It accepts Z3 expressions as input and converts them to PySMT format internally.
    """

    def __init__(self, solver_name: Optional[str] = None) -> None:
        """
        Initialize the PySMT-based AllSMT solver.

        Args:
            solver_name: Optional name of the specific PySMT solver to use
        """
        self._models: List[PySMTModel] = []
        self._model_count: int = 0
        self._solver_name: Optional[str] = solver_name
        self._pysmt_vars: List[FNode] = []
        self._model_limit_reached: bool = False

    def solve(self, expr: ExprRef, keys: List[ExprRef], model_limit: int = 100) -> List[PySMTModel]:
        """
        Enumerate all satisfying models for the given expression over the specified keys.

        Args:
            expr: The Z3 expression/formula to solve
            keys: The Z3 variables to track in the models (not used directly, but kept for API compatibility)
            model_limit: Maximum number of models to generate (default: 100)

        Returns:
            List of PySMT models satisfying the expression
        """
        # Convert Z3 formula to PySMT
        z3_formula = z3.And(expr)
        self._pysmt_vars, pysmt_formula = Z3ToPySMTConverter.convert(z3_formula)
        target_logic = get_logic(pysmt_formula)

        # Reset model storage
        self._models = []
        self._model_count = 0
        self._model_limit_reached = False

        try:
            with Solver(logic=target_logic, name=self._solver_name) as solver:
                solver.add_assertion(pysmt_formula)

                while solver.solve():
                    # Create a model with variable assignments
                    model: PySMTModel = {}
                    for var in self._pysmt_vars:
                        model[var] = solver.get_value(var)

                    self._model_count += 1
                    self._models.append(model)

                    # Check if we've reached the model limit
                    if self._model_count >= model_limit:
                        self._model_limit_reached = True
                        break

                    # Add constraint to find different model in next iteration
                    block_model = []
                    for var in self._pysmt_vars:
                        block_model.append(Not(EqualsOrIff(var, model[var])))
                    solver.add_assertion(Or(block_model))

            return self._models

        except Exception as e:
            raise Exception(f"Error during model sampling: {str(e)}")

    def get_model_count(self) -> int:
        """
        Get the number of models found in the last solve call.

        Returns:
            int: The number of models
        """
        return self._model_count

    @property
    def models(self) -> List[PySMTModel]:
        """
        Get all models found in the last solve call.

        Returns:
            List of PySMT models
        """
        return self._models

    def print_models(self, verbose: bool = False) -> None:
        """
        Print all models found in the last solve call.

        Args:
            verbose: Whether to print detailed information about each model
        """
        if not self._models:
            print("No models found.")
            return

        for i, model in enumerate(self._models):
            if verbose:
                print(f"Model {i + 1}:")
                for var, value in model.items():
                    print(f"  {var} = {value}")
            else:
                print(f"Model {i + 1}: {model}")

        if self._model_limit_reached:
            print(f"Model limit reached. Found {self._model_count} models (there may be more).")
        else:
            print(f"Total number of models: {self._model_count}")


def demo() -> None:
    """Demonstrate the usage of the PySMT-based AllSMT solver with Z3 input."""
    from z3 import Ints, Bools, And, Or, Not

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

    # Create solver and solve with a model limit
    solver = PySMTAllSMTSolver()
    solver.solve(expr, [a, b, x, y], model_limit=10)
    solver.print_models(verbose=True)


if __name__ == "__main__":
    demo()
