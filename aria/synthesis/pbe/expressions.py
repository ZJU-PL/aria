"""Expression types for Version Space Algebra.

This module defines the basic expression types that can represent programs
in different theories (LIA, BV, String).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Union
from enum import Enum


class Theory(Enum):
    """Supported theories for expressions."""
    LIA = "lia"      # Linear Integer Arithmetic
    BV = "bv"        # BitVectors
    STRING = "string"  # Strings


class Expression(ABC):
    """Abstract base class for all expressions."""

    def __init__(self, theory: Theory):
        self.theory = theory

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        """Evaluate the expression given a variable assignment."""
        pass

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Get all variable names used in the expression."""
        pass


class Variable(Expression):
    """Variable expression."""

    def __init__(self, name: str, theory: Theory, sort: str = None):
        super().__init__(theory)
        self.name = name
        self.sort = sort or name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name and self.theory == other.theory

    def __hash__(self) -> int:
        return hash((self.name, self.theory))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        return assignment.get(self.name)

    def get_variables(self) -> Set[str]:
        return {self.name}


class Constant(Expression):
    """Constant expression."""

    def __init__(self, value: Any, theory: Theory):
        super().__init__(theory)
        self.value = value

    def __str__(self) -> str:
        if self.theory == Theory.STRING:
            return f'"{self.value}"'
        elif self.theory == Theory.BV:
            return f"{self.value}bv"
        else:
            return str(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, Constant) and self.value == other.value and self.theory == other.theory

    def __hash__(self) -> int:
        return hash((self.value, self.theory))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        return self.value

    def get_variables(self) -> Set[str]:
        return set()


class BinaryOp(Enum):
    """Binary operations."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"
    CONCAT = "++"  # String concatenation
    BVAND = "&"    # Bitwise AND
    BVOR = "|"     # Bitwise OR
    BVXOR = "^"    # Bitwise XOR
    BVSLL = "<<"   # Bitwise shift left logical
    BVSLR = ">>"   # Bitwise shift right logical
    BVSRA = ">>>"  # Bitwise shift right arithmetic


class UnaryOp(Enum):
    """Unary operations."""
    NEG = "-"
    NOT = "!"
    BVNOT = "~"    # Bitwise NOT
    LENGTH = "len"  # String length


class BinaryExpr(Expression):
    """Binary expression."""

    def __init__(self, left: Expression, op: BinaryOp, right: Expression):
        super().__init__(left.theory)
        self.left = left
        self.op = op
        self.right = right

        # Validate theory compatibility
        if left.theory != right.theory:
            raise ValueError(f"Theory mismatch: {left.theory} vs {right.theory}")

    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, BinaryExpr) and
                self.left == other.left and
                self.op == other.op and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash((self.left, self.op, self.right))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        left_val = self.left.evaluate(assignment)
        right_val = self.right.evaluate(assignment)

        if self.theory == Theory.LIA:
            if self.op == BinaryOp.ADD:
                return left_val + right_val
            elif self.op == BinaryOp.SUB:
                return left_val - right_val
            elif self.op == BinaryOp.MUL:
                return left_val * right_val
            elif self.op == BinaryOp.DIV:
                return left_val // right_val if isinstance(left_val, int) and isinstance(right_val, int) else 0
            elif self.op == BinaryOp.MOD:
                return left_val % right_val if right_val != 0 else 0
            elif self.op == BinaryOp.EQ:
                return left_val == right_val
            elif self.op == BinaryOp.NEQ:
                return left_val != right_val
            elif self.op == BinaryOp.LT:
                return left_val < right_val
            elif self.op == BinaryOp.LE:
                return left_val <= right_val
            elif self.op == BinaryOp.GT:
                return left_val > right_val
            elif self.op == BinaryOp.GE:
                return left_val >= right_val

        elif self.theory == Theory.BV:
            if self.op == BinaryOp.BVAND:
                return left_val & right_val
            elif self.op == BinaryOp.BVOR:
                return left_val | right_val
            elif self.op == BinaryOp.BVXOR:
                return left_val ^ right_val
            elif self.op == BinaryOp.BVSLL:
                return left_val << right_val
            elif self.op == BinaryOp.BVSLR:
                return left_val >> right_val
            elif self.op == BinaryOp.BVSRA:
                return left_val >> right_val  # In Python, >> is arithmetic shift for positive numbers

        elif self.theory == Theory.STRING:
            if self.op == BinaryOp.CONCAT:
                return str(left_val) + str(right_val)
            elif self.op == BinaryOp.EQ:
                return str(left_val) == str(right_val)

        return False

    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()


class UnaryExpr(Expression):
    """Unary expression."""

    def __init__(self, op: UnaryOp, operand: Expression):
        super().__init__(operand.theory)
        self.op = op
        self.operand = operand

    def __str__(self) -> str:
        return f"{self.op.value}({self.operand})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, UnaryExpr) and
                self.op == other.op and
                self.operand == other.operand)

    def __hash__(self) -> int:
        return hash((self.op, self.operand))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        operand_val = self.operand.evaluate(assignment)

        if self.theory == Theory.LIA:
            if self.op == UnaryOp.NEG:
                return -operand_val

        elif self.theory == Theory.STRING:
            if self.op == UnaryOp.LENGTH:
                return len(str(operand_val))

        return operand_val

    def get_variables(self) -> Set[str]:
        return self.operand.get_variables()


# Convenience functions for creating expressions
def var(name: str, theory: Theory, sort: str = None) -> Variable:
    """Create a variable expression."""
    return Variable(name, theory, sort)


def const(value: Any, theory: Theory) -> Constant:
    """Create a constant expression."""
    return Constant(value, theory)


def add(left: Expression, right: Expression) -> BinaryExpr:
    """Create an addition expression."""
    return BinaryExpr(left, BinaryOp.ADD, right)


def sub(left: Expression, right: Expression) -> BinaryExpr:
    """Create a subtraction expression."""
    return BinaryExpr(left, BinaryOp.SUB, right)


def mul(left: Expression, right: Expression) -> BinaryExpr:
    """Create a multiplication expression."""
    return BinaryExpr(left, BinaryOp.MUL, right)


def eq(left: Expression, right: Expression) -> BinaryExpr:
    """Create an equality expression."""
    return BinaryExpr(left, BinaryOp.EQ, right)


def lt(left: Expression, right: Expression) -> BinaryExpr:
    """Create a less-than expression."""
    return BinaryExpr(left, BinaryOp.LT, right)


def concat(left: Expression, right: Expression) -> BinaryExpr:
    """Create a string concatenation expression."""
    return BinaryExpr(left, BinaryOp.CONCAT, right)


def bv_and(left: Expression, right: Expression) -> BinaryExpr:
    """Create a bitwise AND expression."""
    return BinaryExpr(left, BinaryOp.BVAND, right)


def length(expr: Expression) -> UnaryExpr:
    """Create a string length expression."""
    return UnaryExpr(UnaryOp.LENGTH, expr)


class IfExpr(Expression):
    """Conditional expression (if-then-else)."""

    def __init__(self, condition: Expression, then_expr: Expression, else_expr: Expression):
        super().__init__(then_expr.theory)
        self.condition = condition
        self.then_expr = then_expr
        self.else_expr = else_expr

        # Validate theory compatibility
        if then_expr.theory != else_expr.theory:
            raise ValueError(f"Theory mismatch in then/else: {then_expr.theory} vs {else_expr.theory}")

    def __str__(self) -> str:
        return f"(if {self.condition} then {self.then_expr} else {self.else_expr})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, IfExpr) and
                self.condition == other.condition and
                self.then_expr == other.then_expr and
                self.else_expr == other.else_expr)

    def __hash__(self) -> int:
        return hash((self.condition, self.then_expr, self.else_expr))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        condition_val = self.condition.evaluate(assignment)

        # Handle different theory conditions
        if self.condition.theory == Theory.LIA:
            # For LIA, condition should be boolean (non-zero = true)
            if condition_val != 0:
                return self.then_expr.evaluate(assignment)
            else:
                return self.else_expr.evaluate(assignment)
        elif self.condition.theory == Theory.BV:
            # For BV, condition should be non-zero
            if condition_val != 0:
                return self.then_expr.evaluate(assignment)
            else:
                return self.else_expr.evaluate(assignment)
        else:
            # Default behavior
            if condition_val:
                return self.then_expr.evaluate(assignment)
            else:
                return self.else_expr.evaluate(assignment)

    def get_variables(self) -> Set[str]:
        return (self.condition.get_variables() |
                self.then_expr.get_variables() |
                self.else_expr.get_variables())


class LoopExpr(Expression):
    """Loop expression (for loops with fixed iterations)."""

    def __init__(self, variable: str, start: Expression, end: Expression,
                 body: Expression, theory: Theory):
        super().__init__(theory)
        self.variable = variable
        self.start = start
        self.end = end
        self.body = body

    def __str__(self) -> str:
        return f"(for {self.variable} from {self.start} to {self.end} do {self.body})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, LoopExpr) and
                self.variable == other.variable and
                self.start == other.start and
                self.end == other.end and
                self.body == other.body)

    def __hash__(self) -> int:
        return hash((self.variable, self.start, self.end, self.body))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        start_val = self.start.evaluate(assignment)
        end_val = self.end.evaluate(assignment)

        if not (isinstance(start_val, int) and isinstance(end_val, int)):
            raise ValueError("Loop bounds must be integers")

        result = None
        for i in range(start_val, end_val + 1):
            # Create new assignment with loop variable
            loop_assignment = assignment.copy()
            loop_assignment[self.variable] = i
            result = self.body.evaluate(loop_assignment)

        return result

    def get_variables(self) -> Set[str]:
        variables = self.start.get_variables() | self.end.get_variables() | self.body.get_variables()
        variables.discard(self.variable)  # Remove loop variable from outer scope
        return variables


class FunctionCallExpr(Expression):
    """Function call expression."""

    def __init__(self, function_name: str, args: List[Expression], theory: Theory):
        super().__init__(theory)
        self.function_name = function_name
        self.args = args

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.function_name}({args_str})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, FunctionCallExpr) and
                self.function_name == other.function_name and
                self.args == other.args)

    def __hash__(self) -> int:
        return hash((self.function_name, tuple(self.args)))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        # For now, implement basic functions
        if self.function_name == "abs" and len(self.args) == 1:
            arg_val = self.args[0].evaluate(assignment)
            if self.theory == Theory.LIA:
                return abs(arg_val)
        elif self.function_name == "min" and len(self.args) == 2:
            arg1_val = self.args[0].evaluate(assignment)
            arg2_val = self.args[1].evaluate(assignment)
            return min(arg1_val, arg2_val)
        elif self.function_name == "max" and len(self.args) == 2:
            arg1_val = self.args[0].evaluate(assignment)
            arg2_val = self.args[1].evaluate(assignment)
            return max(arg1_val, arg2_val)
        elif self.function_name == "str_substring" and len(self.args) == 3:
            # str_substring(str, start, length)
            s = str(self.args[0].evaluate(assignment))
            start = int(self.args[1].evaluate(assignment))
            length = int(self.args[2].evaluate(assignment))
            return s[start:start+length]
        elif self.function_name == "str_indexof" and len(self.args) == 2:
            # str_indexof(str, substr)
            s = str(self.args[0].evaluate(assignment))
            substr = str(self.args[1].evaluate(assignment))
            return s.find(substr)

        raise ValueError(f"Unknown function: {self.function_name}")

    def get_variables(self) -> Set[str]:
        variables = set()
        for arg in self.args:
            variables.update(arg.get_variables())
        return variables


# Additional convenience functions
def if_expr(condition: Expression, then_expr: Expression, else_expr: Expression) -> IfExpr:
    """Create a conditional expression."""
    return IfExpr(condition, then_expr, else_expr)


def for_loop(variable: str, start: Expression, end: Expression,
             body: Expression, theory: Theory) -> LoopExpr:
    """Create a for loop expression."""
    return LoopExpr(variable, start, end, body, theory)


def func_call(function_name: str, args: List[Expression], theory: Theory) -> FunctionCallExpr:
    """Create a function call expression."""
    return FunctionCallExpr(function_name, args, theory)
