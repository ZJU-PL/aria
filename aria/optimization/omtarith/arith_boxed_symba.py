"""
The Symba algorithm for boxed optimization (multiple independent objectives) of linear arithmetic.
- POPL 14. Symbolic Optimization with SMT Solvers.
"""

import z3
from typing import List, Optional

def arith_boxed_symba(
    fml: z3.ExprRef,
    objs: List[z3.ExprRef],
    minimize: bool,
    solver_name: str
) -> Optional[List[z3.ExprRef]]:
    """
    The Symba algorithm for boxed optimization (multiple independent objectives) of linear arithmetic.

    This implements the compact linear search algorithm from POPL 14 for multiple objectives
    in boxed optimization, where each objective is optimized independently.

    Args:
        fml: The formula representing constraints
        objs: List of objective expressions to optimize
        minimize: True to minimize all objectives, False to maximize all objectives
        solver_name: Name of the SMT solver to use (currently supports "z3")

    Returns:
        List of optimal values for each objective, or None if unsatisfiable
    """
    if solver_name != "z3":
        raise ValueError(f"Unsupported solver: {solver_name}")

    if not objs:
        return []

    solver = z3.Solver()
    solver.add(fml)

    # Check if the formula is satisfiable
    if solver.check() != z3.sat:
        return None

    # Get initial values for all objectives
    model = solver.model()
    current_values = [model.eval(obj, model_completion=True) for obj in objs]

    # Compact linear search: try to improve at least one objective in each iteration
    while True:
        # Create disjunction: at least one objective can be improved
        improvement_constraint = z3.BoolVal(False)
        for i in range(len(objs)):
            if minimize:
                # Try to find smaller values
                improvement_constraint = z3.Or(improvement_constraint, objs[i] < current_values[i])
            else:
                # Try to find larger values
                improvement_constraint = z3.Or(improvement_constraint, objs[i] > current_values[i])

        # Check if any objective can be improved
        if solver.check(improvement_constraint) == z3.unsat:
            # No objective can be improved further
            break

        # Found a better solution - update all objective values
        model = solver.model()
        for i in range(len(objs)):
            new_value = model.eval(objs[i], model_completion=True)
            # Update if this objective improved (or stayed same)
            try:
                if minimize:
                    if (z3.is_int_value(new_value) and z3.is_int_value(current_values[i]) and
                            new_value.as_long() <= current_values[i].as_long()):
                        current_values[i] = new_value
                    elif (z3.is_real_value(new_value) and z3.is_real_value(current_values[i])):
                        # For real values, compare as strings converted to floats
                        new_val = float(str(new_value))
                        curr_val = float(str(current_values[i]))
                        if new_val <= curr_val:
                            current_values[i] = new_value
                else:
                    if (z3.is_int_value(new_value) and z3.is_int_value(current_values[i]) and
                            new_value.as_long() >= current_values[i].as_long()):
                        current_values[i] = new_value
                    elif (z3.is_real_value(new_value) and z3.is_real_value(current_values[i])):
                        # For real values, compare as strings converted to floats
                        new_val = float(str(new_value))
                        curr_val = float(str(current_values[i]))
                        if new_val >= curr_val:
                            current_values[i] = new_value
            except (AttributeError, ValueError, TypeError):
                # If comparison fails, skip this update
                continue

    return current_values
