"""Expression types for typed program synthesis by example."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Theory(Enum):
    """Supported synthesis theories."""

    LIA = "lia"
    BV = "bv"
    STRING = "string"


class ValueType(Enum):
    """Result sorts for expressions."""

    INT = "int"
    BOOL = "bool"
    STRING = "string"
    BV = "bv"


class Expression(ABC):
    """Abstract base class for all expressions."""

    def __init__(
        self,
        theory: Theory,
        value_type: ValueType,
        bitwidth: Optional[int] = None,
    ):
        self.theory = theory
        self.value_type = value_type
        self.bitwidth = bitwidth

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the expression."""

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Check equality with another expression."""

    @abstractmethod
    def __hash__(self) -> int:
        """Return a stable hash."""

    @abstractmethod
    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        """Evaluate the expression given a variable assignment."""

    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Get all variable names used in the expression."""

    @abstractmethod
    def structural_cost(self) -> int:
        """Return a simple size-based ranking cost."""


class Variable(Expression):
    """Variable expression."""

    def __init__(
        self,
        name: str,
        theory: Theory,
        value_type: Optional[ValueType] = None,
        bitwidth: Optional[int] = None,
    ):
        inferred_type = value_type or default_value_type_for_theory(theory)
        super().__init__(theory, inferred_type, bitwidth=bitwidth)
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Variable)
            and self.name == other.name
            and self.theory == other.theory
            and self.value_type == other.value_type
            and self.bitwidth == other.bitwidth
        )

    def __hash__(self) -> int:
        return hash((self.name, self.theory, self.value_type, self.bitwidth))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        return assignment[self.name]

    def get_variables(self) -> Set[str]:
        return {self.name}

    def structural_cost(self) -> int:
        return 1


class Constant(Expression):
    """Constant expression."""

    def __init__(
        self,
        value: Any,
        theory: Theory,
        value_type: Optional[ValueType] = None,
        bitwidth: Optional[int] = None,
    ):
        inferred_type = value_type or infer_value_type(value, theory)
        super().__init__(theory, inferred_type, bitwidth=bitwidth)
        self.value = value

    def __str__(self) -> str:
        if self.value_type == ValueType.STRING:
            return f'"{self.value}"'
        if self.value_type == ValueType.BV:
            return f"{self.value}bv"
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Constant)
            and self.value == other.value
            and self.theory == other.theory
            and self.value_type == other.value_type
            and self.bitwidth == other.bitwidth
        )

    def __hash__(self) -> int:
        return hash(
            (self.value, self.theory, self.value_type, self.bitwidth)
        )

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        del assignment
        return self.value

    def get_variables(self) -> Set[str]:
        return set()

    def structural_cost(self) -> int:
        return 1


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
    CONCAT = "++"
    BVAND = "&"
    BVOR = "|"
    BVXOR = "^"
    BVSLL = "<<"
    BVSLR = ">>"
    BVSRA = ">>>"


class UnaryOp(Enum):
    """Unary operations."""

    NEG = "-"
    NOT = "!"
    BVNOT = "~"
    LENGTH = "len"


class BinaryExpr(Expression):
    """Binary expression."""

    def __init__(self, left: Expression, op: BinaryOp, right: Expression):
        if left.theory != right.theory:
            raise ValueError(f"Theory mismatch: {left.theory} vs {right.theory}")

        value_type = infer_binary_result_type(left, op, right)
        bitwidth = left.bitwidth if value_type == ValueType.BV else None
        super().__init__(left.theory, value_type, bitwidth=bitwidth)
        self.left = left
        self.op = op
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} {self.op.value} {self.right})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, BinaryExpr)
            and self.left == other.left
            and self.op == other.op
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.left, self.op, self.right))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        left_val = self.left.evaluate(assignment)
        right_val = self.right.evaluate(assignment)

        if self.op == BinaryOp.ADD:
            return left_val + right_val
        if self.op == BinaryOp.SUB:
            return left_val - right_val
        if self.op == BinaryOp.MUL:
            return left_val * right_val
        if self.op == BinaryOp.DIV:
            if right_val == 0:
                raise ZeroDivisionError("division by zero")
            return left_val // right_val
        if self.op == BinaryOp.MOD:
            if right_val == 0:
                raise ZeroDivisionError("modulo by zero")
            return left_val % right_val
        if self.op == BinaryOp.EQ:
            return left_val == right_val
        if self.op == BinaryOp.NEQ:
            return left_val != right_val
        if self.op == BinaryOp.LT:
            return left_val < right_val
        if self.op == BinaryOp.LE:
            return left_val <= right_val
        if self.op == BinaryOp.GT:
            return left_val > right_val
        if self.op == BinaryOp.GE:
            return left_val >= right_val
        if self.op == BinaryOp.AND:
            return bool(left_val) and bool(right_val)
        if self.op == BinaryOp.OR:
            return bool(left_val) or bool(right_val)
        if self.op == BinaryOp.CONCAT:
            return str(left_val) + str(right_val)
        if self.op == BinaryOp.BVAND:
            return left_val & right_val
        if self.op == BinaryOp.BVOR:
            return left_val | right_val
        if self.op == BinaryOp.BVXOR:
            return left_val ^ right_val
        if self.op == BinaryOp.BVSLL:
            return left_val << right_val
        if self.op == BinaryOp.BVSLR:
            return left_val >> right_val
        if self.op == BinaryOp.BVSRA:
            return left_val >> right_val

        raise ValueError(f"Unsupported operator: {self.op}")

    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()

    def structural_cost(self) -> int:
        return 1 + self.left.structural_cost() + self.right.structural_cost()


class UnaryExpr(Expression):
    """Unary expression."""

    def __init__(self, op: UnaryOp, operand: Expression):
        value_type = infer_unary_result_type(op, operand)
        super().__init__(operand.theory, value_type, bitwidth=operand.bitwidth)
        self.op = op
        self.operand = operand

    def __str__(self) -> str:
        return f"{self.op.value}({self.operand})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, UnaryExpr)
            and self.op == other.op
            and self.operand == other.operand
        )

    def __hash__(self) -> int:
        return hash((self.op, self.operand))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        operand_val = self.operand.evaluate(assignment)

        if self.op == UnaryOp.NEG:
            return -operand_val
        if self.op == UnaryOp.NOT:
            return not operand_val
        if self.op == UnaryOp.BVNOT:
            return ~operand_val
        if self.op == UnaryOp.LENGTH:
            return len(str(operand_val))

        raise ValueError(f"Unsupported operator: {self.op}")

    def get_variables(self) -> Set[str]:
        return self.operand.get_variables()

    def structural_cost(self) -> int:
        return 1 + self.operand.structural_cost()


class IfExpr(Expression):
    """Conditional expression (if-then-else)."""

    def __init__(
        self, condition: Expression, then_expr: Expression, else_expr: Expression
    ):
        if condition.value_type != ValueType.BOOL:
            raise ValueError("If expression condition must be boolean")
        if then_expr.theory != else_expr.theory:
            raise ValueError(
                f"Theory mismatch in then/else: {then_expr.theory} vs "
                f"{else_expr.theory}"
            )
        if then_expr.value_type != else_expr.value_type:
            raise ValueError(
                "If expression branches must have the same result type"
            )

        super().__init__(
            then_expr.theory,
            then_expr.value_type,
            bitwidth=then_expr.bitwidth,
        )
        self.condition = condition
        self.then_expr = then_expr
        self.else_expr = else_expr

    def __str__(self) -> str:
        return f"(if {self.condition} then {self.then_expr} else {self.else_expr})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, IfExpr)
            and self.condition == other.condition
            and self.then_expr == other.then_expr
            and self.else_expr == other.else_expr
        )

    def __hash__(self) -> int:
        return hash((self.condition, self.then_expr, self.else_expr))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        if self.condition.evaluate(assignment):
            return self.then_expr.evaluate(assignment)
        return self.else_expr.evaluate(assignment)

    def get_variables(self) -> Set[str]:
        return (
            self.condition.get_variables()
            | self.then_expr.get_variables()
            | self.else_expr.get_variables()
        )

    def structural_cost(self) -> int:
        return (
            1
            + self.condition.structural_cost()
            + self.then_expr.structural_cost()
            + self.else_expr.structural_cost()
        )


class LoopExpr(Expression):
    """Loop expression (for loops with fixed iterations)."""

    def __init__(
        self,
        variable: str,
        start: Expression,
        end: Expression,
        body: Expression,
        theory: Theory,
    ):
        if start.value_type != ValueType.INT or end.value_type != ValueType.INT:
            raise ValueError("Loop bounds must be integer expressions")

        super().__init__(theory, body.value_type, bitwidth=body.bitwidth)
        self.variable = variable
        self.start = start
        self.end = end
        self.body = body

    def __str__(self) -> str:
        return f"(for {self.variable} from {self.start} to {self.end} do {self.body})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, LoopExpr)
            and self.variable == other.variable
            and self.start == other.start
            and self.end == other.end
            and self.body == other.body
        )

    def __hash__(self) -> int:
        return hash((self.variable, self.start, self.end, self.body))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        start_val = self.start.evaluate(assignment)
        end_val = self.end.evaluate(assignment)

        if not isinstance(start_val, int) or not isinstance(end_val, int):
            raise ValueError("Loop bounds must evaluate to integers")

        result = None
        for index in range(start_val, end_val + 1):
            loop_assignment = assignment.copy()
            loop_assignment[self.variable] = index
            result = self.body.evaluate(loop_assignment)

        return result

    def get_variables(self) -> Set[str]:
        variables = (
            self.start.get_variables()
            | self.end.get_variables()
            | self.body.get_variables()
        )
        variables.discard(self.variable)
        return variables

    def structural_cost(self) -> int:
        return (
            1
            + self.start.structural_cost()
            + self.end.structural_cost()
            + self.body.structural_cost()
        )


class FunctionCallExpr(Expression):
    """Function call expression."""

    def __init__(self, function_name: str, args: List[Expression], theory: Theory):
        value_type = infer_function_result_type(function_name)
        super().__init__(theory, value_type)
        self.function_name = function_name
        self.args = args

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.function_name}({args_str})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, FunctionCallExpr)
            and self.function_name == other.function_name
            and self.args == other.args
        )

    def __hash__(self) -> int:
        return hash((self.function_name, tuple(self.args)))

    def evaluate(self, assignment: Dict[str, Any]) -> Any:
        if self.function_name == "abs" and len(self.args) == 1:
            return abs(self.args[0].evaluate(assignment))
        if self.function_name == "min" and len(self.args) == 2:
            return min(
                self.args[0].evaluate(assignment),
                self.args[1].evaluate(assignment),
            )
        if self.function_name == "max" and len(self.args) == 2:
            return max(
                self.args[0].evaluate(assignment),
                self.args[1].evaluate(assignment),
            )
        if self.function_name == "str_substring" and len(self.args) == 3:
            string_value = str(self.args[0].evaluate(assignment))
            start_idx = max(0, int(self.args[1].evaluate(assignment)))
            substr_length = max(0, int(self.args[2].evaluate(assignment)))
            return string_value[start_idx : start_idx + substr_length]
        if self.function_name == "str_indexof" and len(self.args) == 2:
            string_value = str(self.args[0].evaluate(assignment))
            substring = str(self.args[1].evaluate(assignment))
            return string_value.find(substring)

        raise ValueError(f"Unknown function: {self.function_name}")

    def get_variables(self) -> Set[str]:
        variables: Set[str] = set()
        for arg in self.args:
            variables.update(arg.get_variables())
        return variables

    def structural_cost(self) -> int:
        return 1 + sum(arg.structural_cost() for arg in self.args)


def default_value_type_for_theory(theory: Theory) -> ValueType:
    """Return the default result type for a theory family."""
    if theory == Theory.STRING:
        return ValueType.STRING
    if theory == Theory.BV:
        return ValueType.BV
    return ValueType.INT


def infer_value_type(value: Any, theory: Theory) -> ValueType:
    """Infer a result type for a Python value."""
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, int):
        if theory == Theory.BV:
            return ValueType.BV
        return ValueType.INT
    raise ValueError(f"Unsupported example value: {value!r}")


def infer_binary_result_type(
    left: Expression, op: BinaryOp, right: Expression
) -> ValueType:
    """Infer the result sort of a binary expression."""
    del right

    if op in {
        BinaryOp.EQ,
        BinaryOp.NEQ,
        BinaryOp.LT,
        BinaryOp.LE,
        BinaryOp.GT,
        BinaryOp.GE,
        BinaryOp.AND,
        BinaryOp.OR,
    }:
        return ValueType.BOOL
    if op == BinaryOp.CONCAT:
        return ValueType.STRING
    if op in {
        BinaryOp.BVAND,
        BinaryOp.BVOR,
        BinaryOp.BVXOR,
        BinaryOp.BVSLL,
        BinaryOp.BVSLR,
        BinaryOp.BVSRA,
    }:
        return ValueType.BV
    return left.value_type


def infer_unary_result_type(op: UnaryOp, operand: Expression) -> ValueType:
    """Infer the result sort of a unary expression."""
    if op == UnaryOp.NOT:
        return ValueType.BOOL
    if op == UnaryOp.LENGTH:
        return ValueType.INT
    return operand.value_type


def infer_function_result_type(function_name: str) -> ValueType:
    """Infer the result sort of a function call."""
    if function_name in {"abs", "min", "max", "str_indexof"}:
        return ValueType.INT
    if function_name == "str_substring":
        return ValueType.STRING
    raise ValueError(f"Unsupported function: {function_name}")


def var(
    name: str,
    theory: Theory,
    value_type: Optional[ValueType] = None,
    bitwidth: Optional[int] = None,
) -> Variable:
    """Create a variable expression."""
    return Variable(name, theory, value_type=value_type, bitwidth=bitwidth)


def const(
    value: Any,
    theory: Theory,
    value_type: Optional[ValueType] = None,
    bitwidth: Optional[int] = None,
) -> Constant:
    """Create a constant expression."""
    return Constant(value, theory, value_type=value_type, bitwidth=bitwidth)


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


def if_expr(
    condition: Expression, then_expr: Expression, else_expr: Expression
) -> IfExpr:
    """Create a conditional expression."""
    return IfExpr(condition, then_expr, else_expr)


def for_loop(
    variable: str, start: Expression, end: Expression, body: Expression, theory: Theory
) -> LoopExpr:
    """Create a for loop expression."""
    return LoopExpr(variable, start, end, body, theory)


def func_call(
    function_name: str, args: List[Expression], theory: Theory
) -> FunctionCallExpr:
    """Create a function call expression."""
    return FunctionCallExpr(function_name, args, theory)
