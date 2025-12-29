"""
MathSAT-based AllSMT solver implementation.

This module provides an implementation of the AllSMT solver using MathSAT.
"""

import tempfile
import subprocess
import os
from typing import List, Any, Dict, Optional
from z3 import Solver, ExprRef

from aria.allsmt.base import AllSMTSolver

# Type alias for MathSAT model (string representation)
MathSATModel = str


class MathSATAllSMTSolver(AllSMTSolver[MathSATModel]):
    """
    MathSAT-based AllSMT solver implementation.

    This class implements the AllSMT solver interface using MathSAT as the underlying solver.
    """

    def __init__(self, mathsat_path: Optional[str] = None) -> None:
        """
        Initialize the MathSAT-based AllSMT solver.

        Args:
            mathsat_path: Optional path to the MathSAT executable
        """
        self._models: List[MathSATModel] = []
        self._model_count: int = 0
        self._mathsat_path: str = mathsat_path or "mathsat"
        self._model_limit_reached: bool = False

        # If mathsat_path is not provided, try to get it from global config
        if not mathsat_path:
            try:
                from aria.global_params import global_config  # pylint: disable=import-outside-toplevel
                path = global_config.get_solver_path("mathsat")
                if path:
                    self._mathsat_path = path
            except (ImportError, AttributeError):
                self._mathsat_path = "mathsat"  # Default to 'mathsat' command

    def _z3_to_smtlib2(self, solver: Solver, keys: List[ExprRef]) -> str:
        """
        Convert Z3 constraints to SMT-LIB2 format with proper all-SAT tracking.

        Args:
            solver: Z3 solver with assertions
            keys: Variables to track in the models

        Returns:
            str: SMT-LIB2 formatted string
        """
        smtlib2 = "(set-logic QF_LIA)\n\n"

        # First declare all original variables
        declarations = [f"(declare-fun {k} () Int)\n" for k in keys]
        smtlib2 += "".join(declarations)
        smtlib2 += "\n"

        # Create Boolean labels for tracking conditions
        bool_labels: List[str] = []
        assertion_labels: Dict[str, Any] = {}
        label_counter = 0

        # Analyze assertions to create Boolean labels
        for assertion in solver.assertions():
            label_name = f"b{label_counter}"
            bool_labels.append(label_name)
            assertion_labels[label_name] = assertion
            smtlib2 += f"(declare-fun {label_name} () Bool)\n"
            label_counter += 1
        smtlib2 += "\n"

        # Add assertions with labels
        for label, assertion in assertion_labels.items():
            smtlib2 += f"(assert (= {label} {assertion.sexpr()}))\n"

        # Add original assertions
        for assertion in solver.assertions():
            smtlib2 += f"(assert {assertion.sexpr()})\n"
        smtlib2 += "\n"

        # Add check-allsat command with Boolean labels
        bool_labels_str = " ".join(bool_labels)
        smtlib2 += f"(check-allsat ({bool_labels_str}))\n"

        return smtlib2

    def solve(
            self, expr: ExprRef, keys: List[ExprRef],
            model_limit: int = 100) -> List[MathSATModel]:
        """
        Enumerate all satisfying models for the given expression over the specified keys.

        Args:
            expr: The Z3 expression/formula to solve
            keys: The Z3 variables to track in the models
            model_limit: Maximum number of models to generate (default: 100)

        Returns:
            List of models satisfying the expression
        """
        # Create Z3 solver and add the expression
        solver = Solver()
        solver.add(expr)

        # Reset model storage
        self._models = []
        self._model_count = 0
        self._model_limit_reached = False

        # Create temporary file for SMT-LIB2 instance
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as tmp:
            tmp_path = tmp.name
            try:
                # Convert constraints to SMT-LIB2 and write to file
                smtlib2_content = self._z3_to_smtlib2(solver, keys)
                tmp.write(smtlib2_content)
                tmp.flush()

                # Call MathSAT with all-sat option
                cmd = [self._mathsat_path, tmp_path]

                # Run MathSAT and capture output
                with subprocess.Popen(
                        cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True) as process:
                    output, _ = process.communicate()

                    # Parse output and store models
                    for line in output.splitlines():
                        if line.startswith("sat"):
                            continue
                        if line.startswith("unsat"):
                            break
                        if line.strip():
                            self._model_count += 1
                            self._models.append(line.strip())

                            # Check if we've reached the model limit
                            if self._model_count >= model_limit:
                                self._model_limit_reached = True
                                break

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        return self._models

    def get_model_count(self) -> int:
        """
        Get the number of models found in the last solve call.

        Returns:
            int: The number of models
        """
        return self._model_count

    @property
    def models(self) -> List[MathSATModel]:
        """
        Get all models found in the last solve call.

        Returns:
            List of models as strings (MathSAT output format)
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
            print(f"Model {i + 1}: {model}")

        if self._model_limit_reached:
            print(f"Model limit reached. Found {self._model_count} models "
                  f"(there may be more).")
        else:
            print(f"Total number of models: {self._model_count}")


def demo() -> None:
    """Demonstrate the usage of the MathSAT-based AllSMT solver."""
    from z3 import Ints, And  # pylint: disable=import-outside-toplevel

    x, y = Ints('x y')
    expr = And(x + y == 5, x > 0, y > 0)

    solver = MathSATAllSMTSolver()
    solver.solve(expr, [x, y], model_limit=10)
    solver.print_models()


if __name__ == "__main__":
    demo()
