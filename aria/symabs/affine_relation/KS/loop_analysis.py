from __future__ import annotations

import ast
import inspect
from typing import Dict, List, Tuple, Any, Optional
import z3

# Import the congruence system for creating constraints
from .congruence_system import CongruenceSystem


class LoopAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python loops for congruence abstraction."""

    def __init__(self):
        self.variables: Dict[str, z3.BoolRef] = {}
        self.constraints: List[z3.BoolRef] = []
        self.loop_variables: Dict[str, List[ast.expr]] = {}
        self.current_scope: List[str] = []

    def analyze_function(self, func) -> Tuple[List[z3.BoolRef], Dict[str, z3.BoolRef]]:
        """Analyze a Python function for loop-based congruence opportunities."""
        source = inspect.getsource(func)

        # Fix indentation issues - remove leading whitespace
        lines = source.split('\n')
        if lines and lines[0].strip() == '':
            lines = lines[1:]  # Remove empty first line
        if lines and not lines[0].startswith('def '):
            # Remove common leading whitespace
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            lines = [line[min_indent:] if line.strip() else '' for line in lines]

        source = '\n'.join(lines)
        tree = ast.parse(source)

        # Extract function name and parameters
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            func_def = tree.body[0]
            self.current_scope = [arg.arg for arg in func_def.args.args]

        self.visit(tree)
        return self.constraints, self.variables

    def visit_For(self, node: ast.For) -> None:
        """Analyze for loops - these are prime candidates for congruence analysis."""
        # Track loop variable
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            self.loop_variables[loop_var] = []

            # Analyze loop body for variable relationships
            old_scope = self.current_scope.copy()
            self.current_scope.append(loop_var)

            for stmt in node.body:
                self.visit(stmt)

            self.current_scope = old_scope

        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Analyze while loops."""
        # Analyze condition for potential congruences
        self.visit(node.test)

        # Analyze loop body
        for stmt in node.body:
            self.visit(stmt)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Analyze assignments for variable relationships."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Create Z3 variable if not exists
                if var_name not in self.variables:
                    self.variables[var_name] = z3.Bool(var_name)

                # Track if this variable is assigned in terms of other variables
                if isinstance(node.value, ast.BinOp):
                    self._analyze_binop_assignment(var_name, node.value)
                elif isinstance(node.value, ast.Name):
                    # Simple variable assignment - track relationships
                    if node.value.id in self.variables:
                        # This creates a potential congruence relationship
                        # For now, just note the relationship
                        pass

        self.generic_visit(node)

    def _analyze_binop_assignment(self, var_name: str, binop: ast.BinOp) -> None:
        """Analyze binary operation assignments for congruence patterns."""
        left = binop.left
        right = binop.right

        # Look for patterns like x = y & z, x = y ^ z, etc.
        if isinstance(binop.op, (ast.BitAnd, ast.BitXor, ast.BitOr, ast.LShift, ast.RShift)):
            if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                left_var = self.variables.get(left.id)
                right_var = self.variables.get(right.id)

                if left_var is not None and right_var is not None:
                    # Create Z3 constraint for the bitwise operation
                    if isinstance(binop.op, ast.BitAnd):
                        # For x = y & z, this creates congruence relationships
                        # The result x has bits that are AND of y and z bits
                        constraint = self.variables[var_name] == z3.And(left_var, right_var)
                        self.constraints.append(constraint)
                    elif isinstance(binop.op, ast.BitXor):
                        # For x = y ^ z, this creates XOR relationships
                        constraint = self.variables[var_name] == z3.Xor(left_var, right_var)
                        self.constraints.append(constraint)
                    # Add more operations as needed

    def visit_Compare(self, node: ast.Compare) -> None:
        """Analyze comparisons that might indicate loop invariants."""
        # Look for comparisons that might be loop invariants
        # For example: i < n, x == 0, etc.
        pass

    def extract_congruence_candidates(self) -> List[Tuple[str, int]]:
        """Extract potential congruence relationships from analyzed code."""
        candidates = []

        # Look for bitwise operations that suggest congruences
        # Look for loop variables that are incremented by constants
        # Look for variables that are ANDed, XORed, etc.

        return candidates


def analyze_python_loop(func) -> Tuple[List[z3.BoolRef], Dict[str, z3.BoolRef]]:
    """Analyze a Python function containing loops for congruence abstraction.

    This function uses AST analysis to identify:
    - Loop variables and their relationships
    - Bitwise operations that create congruences
    - Loop invariants that might be expressible as congruences

    Args:
        func: A Python function containing loops to analyze

    Returns:
        Tuple of (constraints, variables) where constraints are Z3 formulas
        representing the loop semantics, and variables maps names to Z3 BoolRefs
    """
    analyzer = LoopAnalyzer()
    return analyzer.analyze_function(func)


def extract_loop_congruences(source_code: str) -> List[Tuple[str, int, int]]:
    """Extract congruence relationships from loop code.

    Args:
        source_code: Python source code containing loops

    Returns:
        List of (variable_name, modulus, expected_value) tuples representing
        discovered congruence relationships
    """
    tree = ast.parse(source_code)
    analyzer = LoopAnalyzer()
    analyzer.visit(tree)

    congruences = []

    # Look for common patterns that create congruences:
    # 1. Variables incremented by 1 in loops (mod 2^k relationships)
    # 2. Bitwise operations
    # 3. Masking operations

    for var_name, expressions in analyzer.loop_variables.items():
        # Check if this looks like a loop counter
        if expressions:
            # This is a simplified analysis - in practice we'd need more
            # sophisticated pattern matching
            congruences.append((var_name, 2, 0))  # Example: variable ≡ 0 (mod 2)

    return congruences


# Example loop patterns from typical bit-twiddling code
def example_parity_loop():
    """Example from the paper: parity computation loop."""
    x = 0
    p = 0
    for i in range(16):
        p = x ^ p
        x = x + 1
    return p


def example_bit_counting():
    """Example: bit counting loop."""
    x = 42
    count = 0
    while x != 0:
        count = count + (x & 1)
        x = x >> 1
    return count
