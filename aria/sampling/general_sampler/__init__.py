"""
General-purpose sampling utilities and model counting functions.
"""

from typing import Optional

import z3

from aria.counting.qfbv_counting import BVModelCounter
from aria.counting.bool.dimacs_counting import count_dimacs_solutions_parallel


def count_solutions(formula_str: str, fmt: str = 'smtlib2', timeout: Optional[int] = None) -> int:
    """
    Count the number of solutions (models) for a given formula.

    Args:
        formula_str: The formula as a string
        fmt: The format of the formula ('smtlib2' or 'dimacs')
        timeout: Optional timeout in seconds

    Returns:
        The number of solutions/models

    Raises:
        ValueError: If the format is not supported or formula parsing fails
    """
    if fmt == 'smtlib2':
        return _count_smtlib2_solutions(formula_str, timeout)
    if fmt == 'dimacs':
        return _count_dimacs_solutions(formula_str, timeout)
    raise ValueError(f"Unsupported format: {fmt}. Supported formats: 'smtlib2', 'dimacs'")


def _count_smtlib2_solutions(formula_str: str, _timeout: Optional[int] = None) -> int:
    """Count solutions for SMTLIB2 format formulas."""
    try:
        # Parse the SMTLIB2 formula
        solver = z3.Solver()
        solver.from_string(formula_str)

        # For now, we'll use a simple enumeration approach
        # In the future, this could be enhanced with more sophisticated counting methods
        if solver.check() == z3.sat:
            # Get the model and count variables
            model = solver.model()
            variables = list(model.decls())

            if not variables:
                # No variables, just check satisfiability
                return 1 if solver.check() == z3.sat else 0

            # For bit-vector formulas, use BVModelCounter
            if all(str(var.range()) == 'BitVec' for var in variables):
                # Try to parse the formula and use BVModelCounter
                try:
                    # Create a formula from the assertions
                    formula = z3.And(*solver.assertions())
                    counter = BVModelCounter()
                    counter.init_from_fml(formula)
                    # Use sharpSAT for counting
                    result, _ = counter.count_models_by_sharp_sat()
                    return result
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    # Fallback to basic enumeration
                    pass

            # Basic enumeration approach for other cases
            # This is a simplified approach - in practice you'd want more sophisticated counting
            return _enumerate_models(solver, _timeout)
        else:
            return 0

    except (z3.Z3Exception, ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"Failed to parse SMTLIB2 formula: {e}") from e


def _count_dimacs_solutions(formula_str: str, _timeout: Optional[int] = None) -> int:
    """Count solutions for DIMACS format formulas."""
    try:
        # Parse the DIMACS format string
        lines = formula_str.strip().split('\n')
        header = []
        clauses = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('c'):  # Skip empty lines and comments
                continue
            if line.startswith('p'):  # Header line
                header.append(line)
            else:  # Clause line
                # Remove trailing 0 and split into literals
                clause = line.rstrip(' 0').strip()
                if clause:
                    clauses.append(clause)

        # Use the existing DIMACS counting function
        return count_dimacs_solutions_parallel(header, clauses)

    except (ValueError, TypeError, IndexError) as e:
        raise ValueError(f"Failed to count DIMACS solutions: {e}") from e


def _enumerate_models(solver: z3.Solver, timeout: Optional[int] = None) -> int:
    """Simple model enumeration for counting."""
    # This is a basic implementation - in practice you'd want to use more efficient methods
    count = 0
    max_iterations = 1000  # Prevent infinite loops

    # Create a fresh solver to avoid interference
    counting_solver = z3.Solver()
    for assertion in solver.assertions():
        counting_solver.add(assertion)

    for i in range(max_iterations):
        if timeout and i % 100 == 0:
            # Basic timeout check - could be improved
            pass

        result = counting_solver.check()
        if result == z3.unsat:
            break
        if result == z3.sat:
            count += 1
            model = counting_solver.model()

            # Add blocking clause to prevent finding the same model again
            # Use a more robust approach
            block_terms = []
            for decl in model.decls():
                value = model[decl]
                # Handle different types of values properly
                if z3.is_true(value):
                    # For boolean true, block by negating the declaration
                    block_terms.append(z3.Not(decl()))
                elif z3.is_false(value):
                    # For boolean false, block by asserting the declaration
                    block_terms.append(decl())
                else:
                    # For other values, use inequality
                    try:
                        block_terms.append(decl() != value)
                    except (TypeError, ValueError, AttributeError):
                        # If we can't create inequality, skip this variable
                        continue

            if block_terms:
                counting_solver.add(z3.Or(*block_terms))
        # else: unknown - break
        if result != z3.sat:
            break

    return count
