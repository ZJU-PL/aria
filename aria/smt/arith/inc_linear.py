"""
Solving non-linear formulas via incremental linearization

This module implements the incremental linearization approach for SMT solving
of non-linear arithmetic and transcendental functions as described in:

"Incremental Linearization: A practical approach to Satisfiability Modulo
Nonlinear Arithmetic and Transcendental Functions" (Invited Paper)
"""

from typing import List, Set, Tuple, Any, Optional, Dict
from z3 import *
import math


def abstract_to_linear(formula: ExprRef) -> Tuple[ExprRef, Dict[ExprRef, ExprRef]]:
    """Abstract non-linear parts to uninterpreted functions

    Returns:
        Tuple of (abstracted_formula, expr_to_uf_map)
        where expr_to_uf_map maps original expressions to their UF replacements
    """
    abstracted_formula: ExprRef = formula
    expr_to_uf_map: Dict[ExprRef, ExprRef] = {}

    for expr in post_order_traverse(formula):
        if is_non_linear(expr):
            # Create unique UF name based on expression structure
            uf_name = f"nl_{abs(hash(str(expr)))}"
            domain = [arg.sort() for arg in expr.children()]
            uf = Function(uf_name, *domain, expr.sort())

            # Create UF application
            uf_app = uf(*expr.children())
            expr_to_uf_map[expr] = uf_app

            # Replace non-linear expression with UF application
            abstracted_formula = substitute(abstracted_formula, (expr, uf_app))

    return abstracted_formula, expr_to_uf_map


def is_non_linear(expr: ExprRef) -> bool:
    """Check if an expression is non-linear"""
    if not is_app(expr):
        return False

    op_kind = expr.decl().kind()
    op_name = str(expr.decl())

    # Non-linear multiplication (multiple variables)
    if op_kind == Z3_OP_MUL:
        args = expr.children()
        # Count non-constant arguments
        non_const_args = [arg for arg in args if not (is_rational_value(arg) or is_int_value(arg))]
        return len(non_const_args) > 1

    # Division is non-linear
    if op_kind == Z3_OP_DIV:
        return True

    # Exponentiation is non-linear
    if op_kind == Z3_OP_POWER:
        return True

    # Transcendental functions are non-linear
    transcendental_funcs = {
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'exp', 'log', 'sqrt', 'sinh', 'cosh', 'tanh',
        'arcsinh', 'arccosh', 'arctanh'
    }
    if op_name in transcendental_funcs:
        return True

    # Check for uninterpreted functions that might be non-linear
    if op_kind == Z3_OP_UNINTERPRETED:
        # We assume uninterpreted functions are non-linear unless proven otherwise
        return True

    return False


def is_var(expr: ExprRef) -> bool:
    """Check if expression is a variable"""
    return is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED


def post_order_traverse(formula: ExprRef) -> List[ExprRef]:
    """Traverse formula in post-order"""
    visited: Set[ExprRef] = set()
    result: List[ExprRef] = []

    def traverse(expr: ExprRef) -> None:
        if expr in visited:
            return
        if is_app(expr):
            for child in expr.children():
                traverse(child)
        visited.add(expr)
        result.append(expr)

    traverse(formula)
    return result


def is_unsat(formula: ExprRef) -> bool:
    solver = Solver()
    solver.add(formula)
    return solver.check() == unsat


def validate_model(formula: ExprRef, model: ModelRef) -> bool:
    """Validate if a model satisfies the formula in the non-linear world"""
    # Substitute model values into the formula
    substituted = substitute_model(formula, model)
    # Check if the substituted formula is valid
    solver = Solver()
    solver.add(Not(substituted))
    return solver.check() == unsat


def substitute_model(formula: ExprRef, model: ModelRef) -> ExprRef:
    """Substitute model values into a formula"""
    substitutions: List[Tuple[ExprRef, ExprRef]] = []
    for decl in model.decls():
        val = model[decl]
        if is_algebraic_value(val):
            # Handle algebraic numbers
            val = val.approx(20)  # Use 20 digits of precision
        if decl.arity() == 0:  # Only substitute constants/variables
            substitutions.append((decl(), val))
    return substitute(formula, substitutions)


def generate_multiplication_refinements(expr: ExprRef, uf: FuncDeclRef, model: ModelRef) -> List[ExprRef]:
    """Generate refinement constraints for multiplication expressions"""
    refinements = []
    x, y = expr.children()

    # Zero constraints: ∀x,y.(x=0 ∨ y=0) ↔ f_mul(x,y)=0
    x_val = substitute_model(x, model)
    y_val = substitute_model(y, model)
    if x_val == 0 or y_val == 0:
        # At least one is zero, so result should be zero
        concrete_val = substitute_model(expr, model)
        if concrete_val != 0:
            refinements.append(uf(x, y) == 0)
    else:
        # Neither is zero, so result should not be zero
        refinements.append(uf(x, y) != 0)

    # Monotonicity constraints for positive y
    x1, x2 = Consts('x1 x2', x.sort())
    refinements.append(ForAll([x1, x2, y],
                              Implies(And(x1 <= x2, y >= 0),
                                      uf(x1, y) <= uf(x2, y))))

    # Additional monotonicity for y >= 0 case
    y1, y2 = Consts('y1 y2', y.sort())
    refinements.append(ForAll([x, y1, y2],
                              Implies(And(y1 <= y2, y1 >= 0, y2 >= 0, x >= 0),
                                      uf(x, y1) <= uf(x, y2))))

    # Tangent plane constraints at current point
    x_val = float(substitute_model(x, model).as_decimal(10))
    y_val = float(substitute_model(y, model).as_decimal(10))

    # Linear approximation: f(x,y) ≈ x*y + x*(y - y_val) + y*(x - x_val)
    # But since we're using UF, we need to add the tangent constraint
    refinements.append(uf(x, y) >= x * y_val + y_val * x - x_val * y_val)
    refinements.append(uf(x, y) <= x * y_val + y_val * x - x_val * y_val + 0.001)  # Small epsilon for floating point

    return refinements


def generate_piecewise_linear_refinements(expr: ExprRef, uf: FuncDeclRef, model: ModelRef) -> List[ExprRef]:
    """Generate piecewise-linear refinement constraints"""
    refinements = []
    op_name = str(expr.decl())

    # Get the argument
    arg = expr.children()[0] if expr.children() else None
    if not arg:
        return refinements

    arg_val = float(substitute_model(arg, model).as_decimal(10))

    # For transcendental functions, add piecewise linear approximations
    if op_name == 'sin':
        # Piecewise linear approximation for sin(x) around x = arg_val
        # Use points at arg_val-π/4, arg_val, arg_val+π/4
        points = [arg_val - math.pi/4, arg_val, arg_val + math.pi/4]
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i + 1]
            y1, y2 = math.sin(x1), math.sin(x2)

            # Linear interpolation constraint
            if y1 != y2:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                # Add constraint: uf(x) ≤ slope*x + intercept + epsilon
                # and uf(x) ≥ slope*x + intercept - epsilon
                # for x in [x1, x2]
                x_var = arg  # We use the actual variable here
                refinements.append(Implies(And(x_var >= x1, x_var <= x2),
                                         uf(x_var) <= slope * x_var + intercept + 0.01))
                refinements.append(Implies(And(x_var >= x1, x_var <= x2),
                                         uf(x_var) >= slope * x_var + intercept - 0.01))

    elif op_name == 'exp':
        # Piecewise linear approximation for exp(x) around x = arg_val
        # Use points at arg_val-1, arg_val, arg_val+1
        points = [max(0, arg_val - 1), arg_val, arg_val + 1]  # Ensure positive for log domain
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i + 1]
            y1, y2 = math.exp(x1), math.exp(x2)

            if y1 != y2 and x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                x_var = arg
                refinements.append(Implies(And(x_var >= x1, x_var <= x2),
                                         uf(x_var) <= slope * x_var + intercept + 0.1))
                refinements.append(Implies(And(x_var >= x1, x_var <= x2),
                                         uf(x_var) >= slope * x_var + intercept - 0.1))

    return refinements


def generate_transcendental_refinements(expr: ExprRef, uf: FuncDeclRef, model: ModelRef) -> List[ExprRef]:
    """Generate refinement constraints for transcendental function expressions"""
    refinements = []
    op_name = str(expr.decl())

    # Get the argument
    arg = expr.children()[0] if expr.children() else None
    if not arg:
        return refinements

    # NTA (NTA) spuriousness check and refinement
    arg_val = float(substitute_model(arg, model).as_decimal(10))

    # Compute Taylor series approximation
    if op_name == 'sin':
        # sin(x) ≈ x - x^3/6 + x^5/120 - ...
        # Use higher order approximation for better precision
        taylor_approx = arg - (arg**3)/6 + (arg**5)/120
    elif op_name == 'cos':
        # cos(x) ≈ 1 - x^2/2 + x^4/24 - ...
        # Use higher order approximation
        taylor_approx = 1 - (arg**2)/2 + (arg**4)/24
    elif op_name == 'exp':
        # exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 + ...
        # Use higher order approximation
        taylor_approx = 1 + arg + (arg**2)/2 + (arg**3)/6 + (arg**4)/24
    elif op_name == 'log':
        # log(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4 + ...
        # Use higher order approximation
        taylor_approx = (arg - 1) - (arg - 1)**2/2 + (arg - 1)**3/3 - (arg - 1)**4/4
    else:
        # For other functions, use first-order approximation
        taylor_approx = arg

    # Add tangent plane constraint at current point
    concrete_val = substitute_model(expr, model)

    # For better precision, add multiple tangent points
    perturbations = [-0.1, 0, 0.1]  # Small perturbations around current point
    for perturb in perturbations:
        perturb_val = arg_val + perturb
        if perturb_val > 0:  # Ensure positive for functions like log
            # Compute function value at perturbed point
            perturb_expr = arg + perturb
            perturb_func_val = math.sin(perturb_val) if op_name == 'sin' else \
                              math.cos(perturb_val) if op_name == 'cos' else \
                              math.exp(perturb_val) if op_name == 'exp' else \
                              math.log(perturb_val) if op_name == 'log' else 0

            # Add tangent constraint at perturbed point
            refinements.append(uf(perturb_expr) == perturb_func_val)

    # Add piecewise linear refinements for better approximation
    refinements.extend(generate_piecewise_linear_refinements(expr, uf, model))

    return refinements


def generate_refinement(formula: ExprRef, model: ModelRef, expr_to_uf_map: Dict[ExprRef, ExprRef]) -> ExprRef:
    """Generate comprehensive refinement constraints based on the current model"""
    refinements: List[ExprRef] = []

    for expr in post_order_traverse(formula):
        if expr in expr_to_uf_map:
            uf_app = expr_to_uf_map[expr]
            uf = uf_app.decl()

            # Add concrete value constraint
            concrete_val = substitute_model(expr, model)
            refinements.append(uf_app == concrete_val)

            # Add specific refinements based on operation type
            if expr.decl().kind() == Z3_OP_MUL:
                refinements.extend(generate_multiplication_refinements(expr, uf, model))
            elif str(expr.decl()) in ['sin', 'cos', 'tan', 'exp', 'log']:
                refinements.extend(generate_transcendental_refinements(expr, uf, model))

    return And(refinements) if refinements else BoolVal(True)


def incremental_linearization(non_linear_formula: ExprRef, max_iterations: int = 100) -> Tuple[str, Optional[ModelRef]]:
    """Solve non-linear formula using incremental linearization

    Args:
        non_linear_formula: The non-linear formula to solve
        max_iterations: Maximum number of refinement iterations

    Returns:
        Tuple of (result, model) where result is "SAT", "UNSAT", or "UNKNOWN"
    """
    # Initial abstraction
    linear_formula, expr_to_uf_map = abstract_to_linear(non_linear_formula)
    solver = Solver()

    # Set solver options for better performance
    solver.set("timeout", 10000)  # 10 second timeout per check

    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check satisfiability of current abstraction
        check_result = solver.check()
        if check_result == unsat:
            return "UNSAT", None
        elif check_result == unknown:
            print(f"Warning: Solver returned unknown at iteration {iteration}")
            return "UNKNOWN", None

        # Get model from abstraction
        model = solver.model()

        # Validate model against original non-linear formula
        if validate_model(non_linear_formula, model):
            return "SAT", model

        # Model is spurious, generate refinement constraints
        refinement = generate_refinement(non_linear_formula, model, expr_to_uf_map)
        if refinement != BoolVal(True):  # Only add non-trivial refinements
            solver.add(refinement)
            print(f"Added refinement at iteration {iteration}")
        else:
            # No more refinements possible
            print(f"No more refinements possible at iteration {iteration}")
            return "UNKNOWN", None

    print(f"Reached maximum iterations ({max_iterations})")
    return "UNKNOWN", None


def run_example_1():
    """Example 1: Simple multiplication constraints"""
    print("=== Example 1: Simple multiplication constraints ===")
    x, y = Reals('x y')
    # x * y > 3 ∧ x + y < 5
    formula = And(x * y > 3, x + y < 5)
    print(f"Formula: {formula}")
    result, model = incremental_linearization(formula)
    print(f"Result: {result}")
    if model:
        print(f"Model: x = {model[x]}, y = {model[y]}")
        print(f"Verification: x*y = {model[x].as_decimal(10)} * {model[y].as_decimal(10)} = {float(model[x].as_decimal(10)) * float(model[y].as_decimal(10))}")
    print()


def run_example_2():
    """Example 2: Transcendental functions"""
    print("=== Example 2: Transcendental functions ===")
    x = Real('x')
    # sin(x) > 0.5 ∧ x < π/2
    formula = And(Sin(x) > 0.5, x < 1.57)  # π/2 ≈ 1.57
    print(f"Formula: {formula}")
    result, model = incremental_linearization(formula)
    print(f"Result: {result}")
    if model:
        print(f"Model: x = {model[x]}")
        print(f"Verification: sin(x) = {math.sin(float(model[x].as_decimal(10)))}")
    print()


def run_example_3():
    """Example 3: Complex non-linear formula"""
    print("=== Example 3: Complex non-linear formula ===")
    x, y = Reals('x y')
    # x² + y² < 1 ∧ x * y > 0.1 ∧ sin(x) > cos(y)
    formula = And(x**2 + y**2 < 1, x * y > 0.1, sin(x) > cos(y))
    print(f"Formula: {formula}")
    result, model = incremental_linearization(formula, max_iterations=50)
    print(f"Result: {result}")
    if model:
        x_val = float(model[x].as_decimal(10))
        y_val = float(model[y].as_decimal(10))
        print(f"Model: x = {model[x]}, y = {model[y]}")
        print(f"Verification:")
        print(f"  x² + y² = {x_val**2 + y_val**2}")
        print(f"  x * y = {x_val * y_val}")
        print(f"  sin(x) = {math.sin(x_val)}")
        print(f"  cos(y) = {math.cos(y_val)}")
    print()


def run_example_4():
    """Example 4: Exponential function"""
    print("=== Example 4: Exponential function ===")
    x = Real('x')
    # exp(x) < 2 ∧ x > 0 ∧ log(x) > -1
    formula = And(exp(x) < 2, x > 0, Log(x) > -1)
    print(f"Formula: {formula}")
    result, model = incremental_linearization(formula)
    print(f"Result: {result}")
    if model:
        x_val = float(model[x].as_decimal(10))
        print(f"Model: x = {model[x]}")
        print(f"Verification:")
        print(f"  exp(x) = {math.exp(x_val)}")
        print(f"  log(x) = {math.log(x_val)}")
    print()


if __name__ == "__main__":
    print("Incremental Linearization SMT Solver Examples")
    print("=" * 50)

    try:
        run_example_1()
        run_example_2()
        run_example_3()
        run_example_4()
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
