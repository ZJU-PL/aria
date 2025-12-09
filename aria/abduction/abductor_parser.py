"""Parse abduction problems from SMT-LIB2 to Z3 expressions."""

from typing import Tuple, Dict, Any
import z3

from aria.abduction.utils import extract_variables_from_smt2
from aria.utils.sexpr import SExprParser


def balance_parentheses(expr: str) -> str:
    """Balance parentheses in expression."""
    open_count = expr.count('(')
    close_count = expr.count(')')
    return expr + ')' * (open_count - close_count) if open_count > close_count else expr


def extract_balanced_expr(text: str, start_idx: int = 0) -> str:
    """Extract balanced expression with matching parentheses."""
    stack = []
    for i in range(start_idx, len(text)):
        if text[i] == '(':
            stack.append(i)
        elif text[i] == ')':
            if stack:
                stack.pop()
                if not stack:
                    return text[start_idx:i+1]
    return text[start_idx:]


def get_sort_str(var: z3.ExprRef) -> str:
    """Get SMT-LIB2 sort string for Z3 variable."""
    if hasattr(var, 'sort'):
        sort = var.sort()
        if z3.is_bv_sort(sort):
            return f"(_ BitVec {sort.size()})"
        elif z3.is_array_sort(sort):
            return f"(Array {sort.domain()} {sort.range()})"
        return str(sort)
    elif isinstance(var, z3.FuncDeclRef):
        domain = [str(var.domain(i)) for i in range(var.arity())]
        return f"({' '.join(domain)}) {var.range()}"
    return "Int"


def parse_smt2_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """Parse SMT-LIB2 expression to Z3."""
    return parse_expr(expr_str, variables)


def extract_assertion(smt2_str: str, start: int) -> Tuple[str, int]:
    """Extract assertion from SMT-LIB2 string."""
    expr_start = smt2_str.find('(', start + 7) or start + 7
    expr = extract_balanced_expr(smt2_str, expr_start)
    next_pos = smt2_str.find(')', expr_start + len(expr)) + 1
    return expr, next_pos


def extract_abduction_goal(smt2_str: str) -> str:
    """Extract abduction goal from SMT-LIB2 string."""
    idx = smt2_str.find('(get-abduct')
    if idx == -1:
        return None
    expr_start = smt2_str.find('(', idx + 10)
    return extract_balanced_expr(smt2_str, expr_start) if expr_start != -1 else None


def parse_abduction_problem(smt2_str: str) -> Tuple[z3.BoolRef, z3.BoolRef, Dict[str, Any]]:
    """Parse abduction problem from SMT-LIB2 to Z3 formulas."""
    variables = extract_variables_from_smt2(smt2_str)

    # Extract preconditions
    assertions = []
    pos = smt2_str.find('(assert')
    while pos != -1:
        expr, next_pos = extract_assertion(smt2_str, pos)
        if expr:
            try:
                assertions.append(parse_expr(expr, variables))
            except Exception as e:
                print(f"Warning: Failed to parse assertion: {expr}. {e}")
        pos = smt2_str.find('(assert', next_pos)

    # Extract goal
    goal_expr = extract_abduction_goal(smt2_str)
    if not goal_expr:
        raise ValueError("No abduction goal found in the input")
    goal = parse_expr(goal_expr, variables)

    # Combine preconditions
    precond = z3.And(*assertions) if len(assertions) > 1 else (assertions[0] if assertions else z3.BoolVal(True))
    return precond, goal, variables


def parse_expr(expr_str: str, variables: Dict[str, Any]) -> z3.ExprRef:
    """Parse SMT-LIB2 expression to Z3."""
    balanced_expr = balance_parentheses(expr_str)
    try:
        s_expr = SExprParser.parse(balanced_expr)
        if s_expr is None:
            raise ValueError("Failed to parse expression: empty result")
        return _convert_sexpr_to_z3(s_expr, variables)
    except SExprParser.ParseError as e:
        raise ValueError(f"S-expression parse error: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse expression: {balanced_expr}. Error: {e}")


def _convert_sexpr_to_z3(sexpr: Any, variables: Dict[str, Any]) -> z3.ExprRef:
    """Convert S-expression to Z3 expression."""
    # Handle atoms
    if isinstance(sexpr, str):
        if sexpr in variables:
            return variables[sexpr]
        elif sexpr in ["true", "false"]:
            return z3.BoolVal(sexpr == "true")
        elif sexpr.startswith('#x'):
            return z3.BitVecVal(int(sexpr[2:], 16), 32)
        elif sexpr.startswith('#b'):
            return z3.BitVecVal(int(sexpr[2:], 2), len(sexpr) - 2)
        return z3.Int(sexpr)

    # Handle numbers
    if isinstance(sexpr, (int, float)):
        return z3.IntVal(sexpr) if isinstance(sexpr, int) else z3.RealVal(sexpr)

    # Handle lists
    if isinstance(sexpr, list) and sexpr:
        op, args = sexpr[0], [_convert_sexpr_to_z3(arg, variables) for arg in sexpr[1:]]

        # Arithmetic
        if op == "+": return sum(args[1:], args[0])
        elif op == "-": return -args[0] if len(args) == 1 else args[0] - sum(args[1:])
        elif op == "*":
            result = args[0]
            for arg in args[1:]:
                result *= arg
            return result
        elif op in ["div", "/"]: return args[0] / args[1]

        # Comparisons
        elif op == "=": return args[0] == args[1]
        elif op == "<": return args[0] < args[1]
        elif op == "<=": return args[0] <= args[1]
        elif op == ">": return args[0] > args[1]
        elif op == ">=": return args[0] >= args[1]

        # Boolean
        elif op == "and": return z3.And(*args)
        elif op == "or": return z3.Or(*args)
        elif op == "not": return z3.Not(args[0])
        elif op == "=>": return z3.Implies(args[0], args[1])

        # Bit-vector operations
        bv_ops = {"bvadd": lambda a, b: a + b, "bvsub": lambda a, b: a - b, "bvmul": lambda a, b: a * b,
                  "bvudiv": z3.UDiv, "bvurem": z3.URem, "bvslt": lambda a, b: a < b}
        if op in bv_ops:
            return bv_ops[op](args[0], args[1])

        # Bit-vector comparisons
        bv_cmp = {"bvult": z3.ULT, "bvule": z3.ULE, "bvugt": z3.UGT, "bvuge": z3.UGE,
                  "bvsle": lambda a, b: a <= b, "bvsgt": lambda a, b: a > b}
        if op in bv_cmp:
            return bv_cmp[op](args[0], args[1])

        # Arrays and control flow
        elif op == "select": return z3.Select(args[0], args[1])
        elif op == "store": return z3.Store(args[0], args[1], args[2])
        elif op == "ite": return z3.If(args[0], args[1], args[2])

        # Function applications
        elif op in variables and isinstance(variables[op], z3.FuncDeclRef):
            return variables[op](*args)

        # Special cases
        elif op == "_" and len(args) >= 2 and str(args[0]) == "BitVec":
            return z3.BitVec("result", args[1].as_long())

        raise ValueError(f"Unsupported operation: {op}")

    raise ValueError(f"Unsupported S-expression: {sexpr}")


def example_int():
    """Integer variables example."""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (declare-fun z () Int)
    (declare-fun w () Int)
    (declare-fun u () Int)
    (declare-fun v () Int)
    (assert (>= x 0))
    (assert (or (>= x 0) (< u v)))
    (get-abduct A (and (>= (+ x y z w u v) 2) (<= (+ x y z w) 3)))
    """

    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("== Integer Example ==")
        print("Variables:", list(vars.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


def example_bv():
    """Bit-vector variables example."""
    smt2_str = """
    (declare-fun x () (_ BitVec 32))
    (declare-fun y () (_ BitVec 32))
    (declare-fun z () (_ BitVec 32))
    (assert (bvuge x #x00000000))
    (assert (bvult y #x00000064))
    (get-abduct A (bvuge (bvadd x y z) #x00000002))
    """

    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("\n== Bit-Vector Example ==")
        print("Variables:", list(vars.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


def example_mixed():
    """Mixed types example."""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () (_ BitVec 32))
    (declare-fun arr () (Array Int Int))
    (declare-fun f (Int Int) Bool)
    (assert (>= x 0))
    (assert (bvult y #x00000064))
    (assert (= (select arr 5) 10))
    (get-abduct A (> x 5))
    """

    try:
        precond, goal, vars = parse_abduction_problem(smt2_str)
        print("\n== Mixed Types Example ==")
        print("Variables:", list(vars.keys()))
        for name, var in vars.items():
            if isinstance(var, z3.FuncDeclRef):
                args = [str(var.domain(i)) for i in range(var.arity())]
                print(f"  {name}: Function({', '.join(args)}) -> {var.range()}")
            else:
                print(f"  {name}: {var.sort()}")
        print("Precondition:", precond)
        print("Goal:", goal)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_int()
    example_bv()
    example_mixed()
