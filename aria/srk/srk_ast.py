"""
Abstract Syntax Tree (AST) for SRK expressions.

This module provides a structured representation of SRK expressions
as abstract syntax trees, enabling structured manipulation and analysis.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, TypeVar, Generic
from dataclasses import dataclass, field
from fractions import Fraction
from abc import ABC, abstractmethod

T = TypeVar('T')
U = TypeVar('U')


class ASTNode(ABC):
    """Base class for AST nodes."""

    @abstractmethod
    def accept(self, visitor: ASTVisitor[T]) -> T:
        """Accept a visitor."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation."""
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Equality check."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Hash function."""
        pass


class ASTVisitor(Generic[T]):
    """Visitor pattern for AST nodes."""

    @abstractmethod
    def visit_variable(self, node: Variable) -> T:
        """Visit variable node."""
        pass

    @abstractmethod
    def visit_constant(self, node: Constant) -> T:
        """Visit constant node."""
        pass

    @abstractmethod
    def visit_binary_op(self, node: BinaryOp) -> T:
        """Visit binary operation node."""
        pass

    @abstractmethod
    def visit_unary_op(self, node: UnaryOp) -> T:
        """Visit unary operation node."""
        pass

    @abstractmethod
    def visit_function_call(self, node: FunctionCall) -> T:
        """Visit function call node."""
        pass

    @abstractmethod
    def visit_quantifier(self, node: Quantifier) -> T:
        """Visit quantifier node."""
        pass

    @abstractmethod
    def visit_conditional(self, node: Conditional) -> T:
        """Visit conditional node."""
        pass


@dataclass(frozen=True)
class Variable(ASTNode):
    """Variable node."""

    name: str
    var_type: str  # "int", "real", "bool"

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_variable(self)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name and self.var_type == other.var_type

    def __hash__(self) -> int:
        return hash((self.name, self.var_type))


@dataclass(frozen=True)
class Constant(ASTNode):
    """Constant node."""

    value: Union[int, float, Fraction, bool]
    const_type: str  # "int", "real", "bool"

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_constant(self)

    def __str__(self) -> str:
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return False
        return self.value == other.value and self.const_type == other.const_type

    def __hash__(self) -> int:
        return hash((self.value, self.const_type))


class BinaryOpType:
    """Binary operation types."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    EQ = "eq"
    LT = "lt"
    LEQ = "leq"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"


class UnaryOpType:
    """Unary operation types."""
    NEG = "neg"
    NOT = "not"
    FLOOR = "floor"
    CEIL = "ceil"


@dataclass(frozen=True)
class BinaryOp(ASTNode):
    """Binary operation node."""

    op_type: str
    left: ASTNode
    right: ASTNode

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_binary_op(self)

    def __str__(self) -> str:
        return f"({self.left} {self.op_type} {self.right})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryOp):
            return False
        return (self.op_type == other.op_type and
                self.left == other.left and
                self.right == other.right)

    def __hash__(self) -> int:
        return hash((self.op_type, self.left, self.right))


@dataclass(frozen=True)
class UnaryOp(ASTNode):
    """Unary operation node."""

    op_type: str
    operand: ASTNode

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_unary_op(self)

    def __str__(self) -> str:
        return f"{self.op_type}({self.operand})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnaryOp):
            return False
        return self.op_type == other.op_type and self.operand == other.operand

    def __hash__(self) -> int:
        return hash((self.op_type, self.operand))


@dataclass(frozen=True)
class FunctionCall(ASTNode):
    """Function call node."""

    function_name: str
    arguments: Tuple[ASTNode, ...]

    def __init__(self, function_name: str, arguments: List[ASTNode]):
        object.__setattr__(self, 'function_name', function_name)
        object.__setattr__(self, 'arguments', tuple(arguments))

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_function_call(self)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.function_name}({args_str})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionCall):
            return False
        return (self.function_name == other.function_name and
                self.arguments == other.arguments)

    def __hash__(self) -> int:
        return hash((self.function_name, self.arguments))


@dataclass(frozen=True)
class Quantifier(ASTNode):
    """Quantifier node."""

    quantifier_type: str  # "forall" or "exists"
    variable: Variable
    body: ASTNode

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_quantifier(self)

    def __str__(self) -> str:
        return f"{self.quantifier_type} {self.variable}. {self.body}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantifier):
            return False
        return (self.quantifier_type == other.quantifier_type and
                self.variable == other.variable and
                self.body == other.body)

    def __hash__(self) -> int:
        return hash((self.quantifier_type, self.variable, self.body))


@dataclass(frozen=True)
class Conditional(ASTNode):
    """Conditional (if-then-else) node."""

    condition: ASTNode
    then_branch: ASTNode
    else_branch: ASTNode

    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_conditional(self)

    def __str__(self) -> str:
        return f"if {self.condition} then {self.then_branch} else {self.else_branch}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Conditional):
            return False
        return (self.condition == other.condition and
                self.then_branch == other.then_branch and
                self.else_branch == other.else_branch)

    def __hash__(self) -> int:
        return hash((self.condition, self.then_branch, self.else_branch))


class ASTBuilder:
    """Builder for constructing AST nodes."""

    def variable(self, name: str, var_type: str = "int") -> Variable:
        """Create a variable node."""
        return Variable(name, var_type)

    def constant(self, value: Union[int, float, Fraction, bool], const_type: str = "int") -> Constant:
        """Create a constant node."""
        return Constant(value, const_type)

    def binary_op(self, op_type: str, left: ASTNode, right: ASTNode) -> BinaryOp:
        """Create a binary operation node."""
        return BinaryOp(op_type, left, right)

    def unary_op(self, op_type: str, operand: ASTNode) -> UnaryOp:
        """Create a unary operation node."""
        return UnaryOp(op_type, operand)

    def function_call(self, function_name: str, arguments: List[ASTNode]) -> FunctionCall:
        """Create a function call node."""
        return FunctionCall(function_name, arguments)

    def quantifier(self, quantifier_type: str, variable: Variable, body: ASTNode) -> Quantifier:
        """Create a quantifier node."""
        return Quantifier(quantifier_type, variable, body)

    def conditional(self, condition: ASTNode, then_branch: ASTNode, else_branch: ASTNode) -> Conditional:
        """Create a conditional node."""
        return Conditional(condition, then_branch, else_branch)


class ASTConverter:
    """Convert between different AST representations."""

    def __init__(self, context: Any):
        self.context = context

    def from_expression(self, expression: Any) -> ASTNode:
        """Convert an expression to AST node."""
        # This would convert from the expression module to AST
        # Placeholder implementation
        return Variable("x", "int")

    def to_expression(self, node: ASTNode) -> Any:
        """Convert AST node to expression."""
        # This would convert from AST to the expression module
        # Placeholder implementation
        return None


class ASTAnalyzer:
    """Analyze AST nodes."""

    def __init__(self):
        pass

    def get_variables(self, node: ASTNode) -> Set[Variable]:
        """Get all variables in an AST."""
        class VariableCollector(ASTVisitor[Set[Variable]]):
            def visit_variable(self, node: Variable) -> Set[Variable]:
                return {node}

            def visit_constant(self, node: Constant) -> Set[Variable]:
                return set()

            def visit_binary_op(self, node: BinaryOp) -> Set[Variable]:
                left_vars = node.left.accept(self)
                right_vars = node.right.accept(self)
                return left_vars | right_vars

            def visit_unary_op(self, node: UnaryOp) -> Set[Variable]:
                return node.operand.accept(self)

            def visit_function_call(self, node: FunctionCall) -> Set[Variable]:
                result = set()
                for arg in node.arguments:
                    result.update(arg.accept(self))
                return result

            def visit_quantifier(self, node: Quantifier) -> Set[Variable]:
                body_vars = node.body.accept(self)
                # Remove the quantified variable
                return body_vars - {node.variable}

            def visit_conditional(self, node: Conditional) -> Set[Variable]:
                condition_vars = node.condition.accept(self)
                then_vars = node.then_branch.accept(self)
                else_vars = node.else_branch.accept(self)
                return condition_vars | then_vars | else_vars

        collector = VariableCollector()
        return node.accept(collector)

    def get_depth(self, node: ASTNode) -> int:
        """Get the depth of an AST."""
        class DepthCalculator(ASTVisitor[int]):
            def visit_variable(self, node: Variable) -> int:
                return 1

            def visit_constant(self, node: Constant) -> int:
                return 1

            def visit_binary_op(self, node: BinaryOp) -> int:
                left_depth = node.left.accept(self)
                right_depth = node.right.accept(self)
                return 1 + max(left_depth, right_depth)

            def visit_unary_op(self, node: UnaryOp) -> int:
                return 1 + node.operand.accept(self)

            def visit_function_call(self, node: FunctionCall) -> int:
                if not node.arguments:
                    return 1
                arg_depths = [arg.accept(self) for arg in node.arguments]
                return 1 + max(arg_depths)

            def visit_quantifier(self, node: Quantifier) -> int:
                return 1 + node.body.accept(self)

            def visit_conditional(self, node: Conditional) -> int:
                cond_depth = node.condition.accept(self)
                then_depth = node.then_branch.accept(self)
                else_depth = node.else_branch.accept(self)
                return 1 + max(cond_depth, then_depth, else_depth)

        calculator = DepthCalculator()
        return node.accept(calculator)

    def get_size(self, node: ASTNode) -> int:
        """Get the size (number of nodes) of an AST."""
        class SizeCalculator(ASTVisitor[int]):
            def visit_variable(self, node: Variable) -> int:
                return 1

            def visit_constant(self, node: Constant) -> int:
                return 1

            def visit_binary_op(self, node: BinaryOp) -> int:
                return 1 + node.left.accept(self) + node.right.accept(self)

            def visit_unary_op(self, node: UnaryOp) -> int:
                return 1 + node.operand.accept(self)

            def visit_function_call(self, node: FunctionCall) -> int:
                return 1 + sum(arg.accept(self) for arg in node.arguments)

            def visit_quantifier(self, node: Quantifier) -> int:
                return 1 + node.body.accept(self)

            def visit_conditional(self, node: Conditional) -> int:
                return (1 + node.condition.accept(self) +
                       node.then_branch.accept(self) +
                       node.else_branch.accept(self))

        calculator = SizeCalculator()
        return node.accept(calculator)


class ASTTransformer:
    """Transform AST nodes."""

    def substitute(self, node: ASTNode, substitutions: Dict[Variable, ASTNode]) -> ASTNode:
        """Substitute variables in AST."""
        class SubstitutionVisitor(ASTVisitor[ASTNode]):
            def __init__(self, substitutions: Dict[Variable, ASTNode]):
                self.substitutions = substitutions

            def visit_variable(self, node: Variable) -> ASTNode:
                return self.substitutions.get(node, node)

            def visit_constant(self, node: Constant) -> ASTNode:
                return node

            def visit_binary_op(self, node: BinaryOp) -> ASTNode:
                left = node.left.accept(self)
                right = node.right.accept(self)
                return BinaryOp(node.op_type, left, right)

            def visit_unary_op(self, node: UnaryOp) -> ASTNode:
                operand = node.operand.accept(self)
                return UnaryOp(node.op_type, operand)

            def visit_function_call(self, node: FunctionCall) -> ASTNode:
                args = [arg.accept(self) for arg in node.arguments]
                return FunctionCall(node.function_name, args)

            def visit_quantifier(self, node: Quantifier) -> ASTNode:
                # Don't substitute the quantified variable in its own body
                body = node.body.accept(self)
                return Quantifier(node.quantifier_type, node.variable, body)

            def visit_conditional(self, node: Conditional) -> ASTNode:
                condition = node.condition.accept(self)
                then_branch = node.then_branch.accept(self)
                else_branch = node.else_branch.accept(self)
                return Conditional(condition, then_branch, else_branch)

        visitor = SubstitutionVisitor(substitutions)
        return node.accept(visitor)

    def rename_variables(self, node: ASTNode, renaming: Dict[str, str]) -> ASTNode:
        """Rename variables in AST."""
        class RenamingVisitor(ASTVisitor[ASTNode]):
            def __init__(self, renaming: Dict[str, str]):
                self.renaming = renaming

            def visit_variable(self, node: Variable) -> ASTNode:
                new_name = self.renaming.get(node.name, node.name)
                return Variable(new_name, node.var_type)

            def visit_constant(self, node: Constant) -> ASTNode:
                return node

            def visit_binary_op(self, node: BinaryOp) -> ASTNode:
                left = node.left.accept(self)
                right = node.right.accept(self)
                return BinaryOp(node.op_type, left, right)

            def visit_unary_op(self, node: UnaryOp) -> ASTNode:
                operand = node.operand.accept(self)
                return UnaryOp(node.op_type, operand)

            def visit_function_call(self, node: FunctionCall) -> ASTNode:
                args = [arg.accept(self) for arg in node.arguments]
                return FunctionCall(node.function_name, args)

            def visit_quantifier(self, node: Quantifier) -> ASTNode:
                # Don't rename the quantified variable
                body = node.body.accept(self)
                return Quantifier(node.quantifier_type, node.variable, body)

            def visit_conditional(self, node: Conditional) -> ASTNode:
                condition = node.condition.accept(self)
                then_branch = node.then_branch.accept(self)
                else_branch = node.else_branch.accept(self)
                return Conditional(condition, then_branch, else_branch)

        visitor = RenamingVisitor(renaming)
        return node.accept(visitor)


class ASTPrinter:
    """Pretty printer for AST nodes."""

    def __init__(self, indent: str = "  "):
        self.indent = indent

    def print_ast(self, node: ASTNode, level: int = 0) -> str:
        """Print AST in a tree format."""
        current_indent = self.indent * level

        if isinstance(node, Variable):
            return f"{current_indent}Variable({node.name}, {node.var_type})"

        elif isinstance(node, Constant):
            return f"{current_indent}Constant({node.value}, {node.const_type})"

        elif isinstance(node, BinaryOp):
            result = f"{current_indent}BinaryOp({node.op_type})\n"
            result += self.print_ast(node.left, level + 1) + "\n"
            result += self.print_ast(node.right, level + 1)
            return result

        elif isinstance(node, UnaryOp):
            result = f"{current_indent}UnaryOp({node.op_type})\n"
            result += self.print_ast(node.operand, level + 1)
            return result

        elif isinstance(node, FunctionCall):
            result = f"{current_indent}FunctionCall({node.function_name})\n"
            for arg in node.arguments:
                result += self.print_ast(arg, level + 1) + "\n"
            return result.rstrip()

        elif isinstance(node, Quantifier):
            result = f"{current_indent}Quantifier({node.quantifier_type})\n"
            result += f"{current_indent}{self.indent}Variable: {node.variable}\n"
            result += self.print_ast(node.body, level + 1)
            return result

        elif isinstance(node, Conditional):
            result = f"{current_indent}Conditional\n"
            result += f"{current_indent}{self.indent}Condition:\n"
            result += self.print_ast(node.condition, level + 2) + "\n"
            result += f"{current_indent}{self.indent}Then:\n"
            result += self.print_ast(node.then_branch, level + 2) + "\n"
            result += f"{current_indent}{self.indent}Else:\n"
            result += self.print_ast(node.else_branch, level + 2)
            return result

        else:
            return f"{current_indent}Unknown({type(node).__name__})"


# Factory functions
def make_variable(name: str, var_type: str = "int") -> Variable:
    """Create a variable node."""
    return Variable(name, var_type)


def make_constant(value: Union[int, float, Fraction, bool], const_type: str = "int") -> Constant:
    """Create a constant node."""
    return Constant(value, const_type)


def make_binary_op(op_type: str, left: ASTNode, right: ASTNode) -> BinaryOp:
    """Create a binary operation node."""
    return BinaryOp(op_type, left, right)


def make_unary_op(op_type: str, operand: ASTNode) -> UnaryOp:
    """Create a unary operation node."""
    return UnaryOp(op_type, operand)


def make_function_call(function_name: str, arguments: List[ASTNode]) -> FunctionCall:
    """Create a function call node."""
    return FunctionCall(function_name, arguments)


def make_quantifier(quantifier_type: str, variable: Variable, body: ASTNode) -> Quantifier:
    """Create a quantifier node."""
    return Quantifier(quantifier_type, variable, body)


def make_conditional(condition: ASTNode, then_branch: ASTNode, else_branch: ASTNode) -> Conditional:
    """Create a conditional node."""
    return Conditional(condition, then_branch, else_branch)


def make_ast_builder() -> ASTBuilder:
    """Create an AST builder."""
    return ASTBuilder()


def make_ast_analyzer() -> ASTAnalyzer:
    """Create an AST analyzer."""
    return ASTAnalyzer()


def make_ast_transformer() -> ASTTransformer:
    """Create an AST transformer."""
    return ASTTransformer()


def make_ast_printer(indent: str = "  ") -> ASTPrinter:
    """Create an AST printer."""
    return ASTPrinter(indent)
