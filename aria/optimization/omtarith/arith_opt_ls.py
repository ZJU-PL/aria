"""
Local search for optimization of linear arithmetic.
"""

import logging
import random

import z3

# Import utility function from the utils module
from aria.utils.z3_expr_utils import get_expr_vars

logger = logging.getLogger(__name__)

def _arith_opt_with_ls_impl(
    fml: z3.ExprRef,
    obj: z3.ExprRef,
    minimize: bool,
    solver_name: str,
    max_iterations: int = 1000
):
    """
    Local search for optimization of linear arithmetic (implementation).

    Args:
        fml: Z3 formula representing constraints
        obj: Z3 expression representing the objective function
        minimize: True to minimize, False to maximize
        solver_name: Name of the SMT solver to use
        max_iterations: Maximum number of local search iterations

    Returns:
        String representation of the optimal value found
    """
    logger.debug("Starting local search optimization with %s", solver_name)
    logger.debug("Direction: %s", 'minimize' if minimize else 'maximize')

    # First check if the constraints are satisfiable
    solver = z3.Solver()
    solver.add(fml)

    if solver.check() != z3.sat:
        logger.warning("Constraints are unsatisfiable")
        return "unsat"

    # Get initial model that satisfies constraints
    current_model = solver.model()
    current_obj_value = current_model.eval(obj, model_completion=True)

    logger.debug("Initial objective value: %s", current_obj_value)

    # Get all variables in the formula and objective
    all_vars = set()
    all_vars.update(get_expr_vars(fml))
    all_vars.update(get_expr_vars(obj))

    # Filter to only arithmetic variables (Int/Real)
    arith_vars = [var for var in all_vars if z3.is_int(var) or z3.is_real(var)]

    logger.debug("Found %d arithmetic variables", len(arith_vars))

    # Initialize best solution
    best_model = current_model
    best_obj_value = current_obj_value

    # Try multiple starting points for better exploration
    restart_attempts = 3

    for restart in range(restart_attempts):
        if restart > 0:
            # Generate a new starting point by randomizing variables
            logger.debug("Restart %d: Trying new starting point", restart)
            temp_solver = z3.Solver()
            temp_solver.add(fml)

            # Add random constraints to get different starting points
            # Limit to avoid over-constraining
            for var in arith_vars[:min(3, len(arith_vars))]:
                if z3.is_int(var):
                    random_val = random.randint(-50, 50)
                    temp_solver.push()
                    temp_solver.add(var == random_val)
                    if temp_solver.check() == z3.sat:
                        current_model = temp_solver.model()
                        temp_solver.pop()
                        break
                    temp_solver.pop()
                elif z3.is_real(var):
                    random_val = random.uniform(-50.0, 50.0)
                    temp_solver.push()
                    temp_solver.add(var == random_val)
                    if temp_solver.check() == z3.sat:
                        current_model = temp_solver.model()
                        temp_solver.pop()
                        break
                    temp_solver.pop()

        # Main local search loop
        improved_overall = True
        iteration = 0
        local_best_value = current_model.eval(obj, model_completion=True)

        while improved_overall and iteration < max_iterations // restart_attempts:
            iteration += 1
            improved_overall = False

            # Randomize the order of variables to explore
            vars_to_try = arith_vars.copy()
            random.shuffle(vars_to_try)

            # Try to improve each variable
            for var in vars_to_try:
                try:
                    # Get current value of the variable
                    current_value = current_model.eval(var, model_completion=True)

                    # Determine step sizes based on variable type and iteration
                    if z3.is_int(var):
                        # Adaptive step sizes - larger steps early, smaller steps later
                        base_steps = [-5, -3, -2, -1, 1, 2, 3, 5]
                        large_steps = [-20, -10, 10, 20] if iteration < 10 else []
                        offsets = base_steps + large_steps

                        try:
                            base_val = current_value.as_long()
                            candidates = [base_val + offset for offset in offsets]
                        except (AttributeError, ValueError, TypeError):
                            # If conversion fails, skip this variable
                            continue
                    else:
                        # For reals, adaptive fractional offsets
                        base_steps = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
                        large_steps = (
                            [-10.0, -5.0, 5.0, 10.0] if iteration < 10 else []
                        )
                        offsets = base_steps + large_steps

                        try:
                            if hasattr(current_value, 'as_fraction'):
                                num, den = current_value.as_fraction()
                                base_val = float(num) / float(den)
                            else:
                                base_val = float(str(current_value))
                            candidates = [
                                base_val + offset for offset in offsets
                            ]
                        except (
                            AttributeError, ValueError, TypeError,
                            ZeroDivisionError
                        ):
                            # If conversion fails, skip this variable
                            continue

                    # Try each candidate value
                    for candidate_value in candidates:
                        # Create a temporary solver to test this assignment
                        temp_solver = z3.Solver()
                        temp_solver.add(fml)

                        # Add assignment for this variable
                        if z3.is_int(var):
                            temp_solver.add(var == int(candidate_value))
                        else:
                            temp_solver.add(var == candidate_value)

                        # Check if this assignment is feasible
                        if temp_solver.check() == z3.sat:
                            new_model = temp_solver.model()
                            new_obj_value = new_model.eval(obj, model_completion=True)

                            # Check if this is an improvement
                            is_better = False
                            try:
                                if minimize:
                                    if (z3.is_int(obj) or
                                            z3.is_int_value(new_obj_value)):
                                        is_better = (
                                            new_obj_value.as_long() <
                                            local_best_value.as_long()
                                        )
                                    else:
                                        # Handle real values
                                        new_val = float(str(new_obj_value))
                                        best_val = float(str(local_best_value))
                                        is_better = new_val < best_val
                                else:
                                    if (z3.is_int(obj) or
                                            z3.is_int_value(new_obj_value)):
                                        is_better = (
                                            new_obj_value.as_long() >
                                            local_best_value.as_long()
                                        )
                                    else:
                                        # Handle real values
                                        new_val = float(str(new_obj_value))
                                        best_val = float(str(local_best_value))
                                        is_better = new_val > best_val
                            except (AttributeError, ValueError, TypeError):
                                # If comparison fails, skip
                                continue

                            if is_better:
                                current_model = new_model
                                local_best_value = new_obj_value
                                improved_overall = True
                                logger.debug(
                                    "Restart %d, Iteration %d: "
                                    "Improved objective to %s",
                                    restart, iteration, local_best_value
                                )
                                break

                    # If we found improvement for this variable, move to next iteration
                    if improved_overall:
                        break

                except Exception as e:
                    logger.debug("Error processing variable %s: %s", var, e)
                    continue

        # Update global best if this restart found a better solution
        try:
            is_global_better = False
            if minimize:
                if (z3.is_int(obj) or
                        z3.is_int_value(local_best_value)):
                    is_global_better = (
                        local_best_value.as_long() <
                        best_obj_value.as_long()
                    )
                else:
                    new_val = float(str(local_best_value))
                    best_val = float(str(best_obj_value))
                    is_global_better = new_val < best_val
            else:
                if (z3.is_int(obj) or
                        z3.is_int_value(local_best_value)):
                    is_global_better = (
                        local_best_value.as_long() >
                        best_obj_value.as_long()
                    )
                else:
                    new_val = float(str(local_best_value))
                    best_val = float(str(best_obj_value))
                    is_global_better = new_val > best_val

            if is_global_better:
                best_model = current_model
                best_obj_value = local_best_value
                logger.debug(
                    "Restart %d: New global best: %s",
                    restart, best_obj_value
                )
        except (AttributeError, ValueError, TypeError):
            # If comparison fails, keep previous best
            pass

    logger.info("Local search completed after %d restarts", restart_attempts)
    logger.info("Final objective value: %s", best_obj_value)

    # Return the optimal value as a string
    return str(best_obj_value)


def arith_opt_with_ls(fml: z3.ExprRef, obj: z3.ExprRef, minimize: bool, solver_name: str):
    """
    Local search for optimization of linear arithmetic.

    This function implements a local search algorithm for optimizing linear arithmetic
    expressions subject to constraints. It uses multiple restarts and adaptive step
    sizes to explore the solution space effectively.

    Algorithm features:
    - Multiple random restarts to escape local optima
    - Adaptive step sizes (larger steps early, smaller steps later)
    - Randomized variable selection order
    - Support for both integer and real variables
    - Robust error handling

    Args:
        fml: Z3 formula representing the constraints
        obj: Z3 expression representing the objective function to optimize
        minimize: True to minimize the objective, False to maximize
        solver_name: Name of the SMT solver to use (currently supports "z3")

    Returns:
        String representation of the optimal value found, or "unsat" if
        the constraints are unsatisfiable

    Example:
        >>> x, y = z3.Ints('x y')
        >>> constraints = z3.And(x >= 0, y >= 0, x + y >= 5)
        >>> objective = x + y
        >>> result = arith_opt_with_ls(
        ...     constraints, objective, minimize=True, solver_name="z3"
        ... )
        >>> print(result)  # Should print "5"
    """
    return _arith_opt_with_ls_impl(fml, obj, minimize, solver_name, max_iterations=1000)


def demo():
    """
    Demonstration of the local search arithmetic optimization.
    """
    import time

    print("=" * 60)
    print("Arithmetic Local Search Optimization Demo")
    print("=" * 60)

    # Example 1: Integer minimization
    print("\n1. Integer Minimization Example")
    print("   Variables: x, y (integers)")
    print("   Constraints: x >= 0, y >= 0, x + y >= 8")
    print("   Objective: minimize x + y")

    x, y = z3.Ints('x y')
    fml = z3.And(x >= 0, y >= 0, x + y >= 8)
    obj = x + y

    start_time = time.time()
    result = arith_opt_with_ls(fml, obj, minimize=True, solver_name="z3")
    end_time = time.time()

    print(f"   Result: {result}")
    print(f"   Time: {end_time - start_time:.4f} seconds")

    # Example 2: Real maximization
    print("\n2. Real Maximization Example")
    print("   Variables: x, y (reals)")
    print("   Constraints: x >= 0, y >= 0, 2*x + 3*y <= 12, x <= 4")
    print("   Objective: maximize x + y")

    x, y = z3.Reals('x y')
    fml = z3.And(x >= 0, y >= 0, 2*x + 3*y <= 12, x <= 4)
    obj = x + y

    start_time = time.time()
    result = arith_opt_with_ls(fml, obj, minimize=False, solver_name="z3")
    end_time = time.time()

    print(f"   Result: {result}")
    print(f"   Time: {end_time - start_time:.4f} seconds")

    # Example 3: Complex integer problem
    print("\n3. Complex Integer Problem")
    print("   Variables: x, y, z (integers)")
    print("   Constraints: x + 2*y + 3*z >= 15, x <= 6, y <= 4, z <= 3, all >= 0")
    print("   Objective: minimize x + y + z")

    x, y, z = z3.Ints('x y z')
    fml = z3.And(x + 2*y + 3*z >= 15, x <= 6, y <= 4, z <= 3,
                 x >= 0, y >= 0, z >= 0)
    obj = x + y + z

    start_time = time.time()
    result = arith_opt_with_ls(fml, obj, minimize=True, solver_name="z3")
    end_time = time.time()

    print(f"   Result: {result}")
    print(f"   Time: {end_time - start_time:.4f} seconds")

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == '__main__':
    demo()
