"""
Parse Monadic Predicate Abstraction Queries

Given a formula and a set of predicates {P1,...,Pn},
decide for each Pi, whether F and Pi is satisfiable or not.
"""

import logging
import re
import sys
from functools import reduce
from typing import List, Any

import z3
from z3 import (  # noqa: F401
    And, Array, ArraySort, ArraySortRef, BitVec, BitVecSort, BitVecSortRef,
    BitVecVal, Bool, BoolSort, BoolVal, Concat, Distinct, Extract, FP, FPVal,
    FPSort, FPSortRef, Function, If, Implies, Int, IntSort, LShR, Not, Or,
    Real, RealSort, Select, SignExt, Solver, Sort, Store, UDiv, UGE, UGT, ULE,
    ULT, URem, ZeroExt, Z3_BOOL_SORT, Z3_INT_SORT, Z3_REAL_SORT, fpAbs, fpAdd,
    fpDiv, fpEQ, fpGEQ, fpGT, fpIsInf, fpIsNaN, fpIsZero, fpLEQ, fpLT, fpMul,
    fpNeg, fpSub, RNE, SRem
)

# print(sys.getrecursionlimit())
sys.setrecursionlimit(200000)  # Adjust the value accordingly

logger = logging.getLogger(__name__)


class MonAbsSMTLIBParser:
    """Parser for Monadic Predicate Abstraction SMT-LIB queries."""

    def __init__(self, **kwargs):
        self.solver = Solver()
        self.variables = {}
        self.functions = {}  # Store declared functions
        self.let_bindings = {} # Store let bindings
        # Stack of constraints for each scope level
        # First list (index 0) contains global constraints
        self.constraints_stack: List[List[Any]] = [[]]

        # Stack of varibles for each scope level
        # For arrays and UFs, we may also need to record the scope of sorts
        self.variables_stack: List[List[str]] = [[]]

        self.check_sat_results = []  # for recording the oracle

        self.logic = kwargs.get('logic', None)

        # "only_parse" mode: do not execute the check-sat commands; just record the cnts
        # if we need to record the oracle, we shoud ignore the option.
        self.only_parse = kwargs.get('only_parse', False)

        # extract precond: z3.ExprRef, cnt_list: List[z3.ExprRef]
        self.flag: bool = False
        self.cnt: z3.ExprRef = z3.BoolVal(True)
        self.precond: z3.ExprRef = z3.BoolVal(True)
        self.cnt_list: List[z3.ExprRef] = []

    def flatten_sort(self, sort_expr):
        """Flatten sort expression appropriately"""
        if isinstance(sort_expr, list):
            # If it's a single-element list containing another list, unwrap it
            if len(sort_expr) == 1 and isinstance(sort_expr[0], list):
                return self.flatten_sort(sort_expr[0])
            # Otherwise, keep the list structure but flatten its elements
            return [self.flatten_sort(x) if isinstance(x, list) else x for x in sort_expr]
        return sort_expr

    def get_sort(self, sort_expr) -> Sort:
        """Parse sort expressions into z3 sorts"""
        # Flatten the sort expression first
        sort_expr = self.flatten_sort(sort_expr)

        if isinstance(sort_expr, str):
            if sort_expr == 'Bool':
                return BoolSort()
            if sort_expr == 'Int':
                return IntSort()
            if sort_expr == 'Real':
                return RealSort()
            raise ValueError(f"Unknown sort: {sort_expr}")
        elif isinstance(sort_expr, list):
            if sort_expr[0] == '_':
                if sort_expr[1] == 'BitVec':
                    return BitVecSort(int(sort_expr[2]))
                elif sort_expr[1] == 'FP' or sort_expr[1] == 'FloatingPoint':
                    return FPSort(int(sort_expr[2]), int(sort_expr[3]))
            elif sort_expr[0] == 'Array':
                domain = self.get_sort(sort_expr[1])
                range_sort = self.get_sort(sort_expr[2])
                return ArraySort(domain, range_sort)
        raise ValueError(f"Invalid sort expression: {sort_expr}")

    def current_scope_level(self) -> int:
        """Return the current scope level (0 is global scope)"""
        return len(self.constraints_stack) - 1

    def add_constraint(self, constraint, expr):
        """Add a constraint to the current scope"""
        self.constraints_stack[-1].append((constraint, expr))

    def get_current_scope_constraints(self):
        """Get constraints in the current scope"""
        return self.constraints_stack[-1]

    def get_all_active_constraints(self):
        """Get all constraints active in current scope (including parent scopes)"""
        all_constraints = []
        for scope in self.constraints_stack:
            all_constraints.extend(scope)
        return all_constraints

    def tokenize(self, s):
        """Tokenize SMT-LIB input string."""
        # Remove comments
        s = re.sub(';.*\n', '\n', s)
        # Add spaces around parentheses
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        # Split into tokens
        return [token for token in s.split() if token]

    def parse_tokens(self, tokens):
        """Parse tokens into expression tree."""
        if not tokens:
            return None

        if tokens[0] == '(':
            expression = []
            tokens.pop(0)  # Remove opening '('
            while tokens and tokens[0] != ')':
                exp = self.parse_tokens(tokens)
                if exp is not None:
                    expression.append(exp)
            if tokens:
                tokens.pop(0)  # Remove closing ')'
            return expression
        return tokens.pop(0)

    def create_variable(self, name: str, sort) -> Any:
        """Create a variable of the specified sort"""
        z3_sort = self.get_sort(sort)

        if isinstance(z3_sort, BitVecSortRef):
            return BitVec(name, z3_sort.size())
        if isinstance(z3_sort, FPSortRef):
            return FP(name, z3_sort)
        if isinstance(z3_sort, ArraySortRef):
            return Array(name, z3_sort.domain(), z3_sort.range())
        if z3_sort.kind() == Z3_BOOL_SORT:
            return Bool(name)
        if z3_sort.kind() == Z3_INT_SORT:
            return Int(name)
        if z3_sort.kind() == Z3_REAL_SORT:
            return Real(name)
        raise ValueError(f"Unsupported sort kind: {z3_sort.kind()}")

    def process_command(self, command):
        """Process a parsed SMT-LIB command."""
        if not isinstance(command, list):
            return

        cmd = command[0]

        if cmd == 'set-logic':
            self.logic = command[1]

        if cmd == 'declare-const':
            name = command[1]
            sort = command[2] if isinstance(command[2], str) else command[2:]
            self.variables[name] = self.create_variable(name, sort)

        elif cmd == 'declare-fun':
            # Handle function declarations
            name = command[1]
            domain_sorts = [self.get_sort(s) for s in command[2]]
            if len(domain_sorts) == 0:
                sort = command[3] if isinstance(command[3], str) else command[3:]
                self.variables[name] = self.create_variable(name, sort)
            else:
                range_sort = self.get_sort(command[3])
                self.functions[name] = Function(name, *domain_sorts, range_sort)

        # elif cmd == 'declare-sort':
            # TODO: also use constant

        if cmd == 'assert':
            expr = self.build_expression(command[1])
            self.solver.add(expr)
            # Store both the original constraint and the built z3 expression
            self.add_constraint(command[1], expr)
            if not self.flag:
                self.precond = z3.And(self.precond, expr)
            if self.flag:
                self.cnt = z3.And(self.cnt, expr)

        if cmd == 'push':
            self.solver.push()
            self.flag = True
            # Create new scope for constraints
            self.constraints_stack.append([])

        if cmd == 'pop':
            if len(self.constraints_stack) <= 1:
                raise ValueError("Cannot pop global scope")
            self.solver.pop()
            self.flag = False
            self.cnt_list.append(self.cnt)
            self.cnt = z3.BoolVal(True)
            # Remove constraints from the current scope
            popped_constraints = self.constraints_stack.pop()
            logger.debug("Popped constraints from scope %s:", len(self.constraints_stack))
            for original, _ in popped_constraints:
                logger.debug("  %s", original)
            # print(f"Popped constraints from scope {len(self.constraints_stack)}:")
            # for original, _ in popped_constraints:
            #    print(f"  {original}")

        if cmd == 'check-sat':
            # if we set a "only parse mode", do not actually check-sat
            if self.only_parse:
                return
            print("---------------------------------------------------")
            print(self.solver)
            result = self.solver.check()
            print(f"check-sat result: {result}")
            print("Current scope constraints:")
            for original, _ in self.get_current_scope_constraints():
                print(f"  {original}")
            self.check_sat_results.append(result)

    def get_default_fp_sort(self):
        """Get default floating point sort."""
        return FPSort(8, 24)  # Single precision

    def parse_special_fp_value(self, value):
        """Parse special floating point values like +oo, -oo, NaN."""
        # This is a placeholder - implement based on actual SMT-LIB format
        raise NotImplementedError("Special FP value parsing not yet implemented")

    def parse_constant(self, value):
        """Parse constants based on the current logic."""
        try:
            if self.logic and 'FP' in self.logic:
                # Handle floating point constants
                if value.startswith('#b'):
                    # Binary format
                    return FPVal(value[2:], self.get_default_fp_sort())
                if value.startswith('('):
                    # Special values like +oo, -oo, NaN
                    return self.parse_special_fp_value(value)
            if self.logic and 'BV' in self.logic:
                # Handle bit-vector constants
                if value.startswith('#b'):
                    return BitVecVal(int(value[2:], 2), len(value[2:]))
                if value.startswith('#x'):
                    return BitVecVal(int(value[2:], 16), len(value[2:]) * 4)

            # Try parsing as regular numeric constant
            return float(value) if '.' in value else int(value)
        except ValueError:
            return value

    def build_expression(self, expr):
        """Build Z3 expression from parsed SMT-LIB expression."""
        if not isinstance(expr, list):
            # Handle constants and variables
            if expr in self.variables:
                return self.variables[expr]
            if expr in self.functions:
                return self.functions[expr]
            if expr in self.let_bindings:
                return self.let_bindings[expr]
            if expr == 'true':
                return z3.BoolVal(True)
            if expr == 'false':
                return z3.BoolVal(False)
            return self.parse_constant(expr)

        op = expr[0]

        if isinstance(op, list):
            flatten_expr = expr[0]
            flatten_expr.append(expr[1])
            return self.build_special_operator(flatten_expr)

        if op == 'let':
            return self.build_let_expression(expr)

        args = [self.build_expression(arg) for arg in expr[1:]]

        # Theory-specific operations
        if op in self.functions:
            # Function application
            return self.functions[op](*args)
        if op == 'select':
            # Array select
            return Select(args[0], args[1])
        if op == 'store':
            # Array store
            return Store(args[0], args[1], args[2])
        if op.startswith('fp.'):
            # Floating point operations
            return self.build_fp_expression(op, args)
        if op.startswith('bv'):
            # Bit-vector operations
            return self.build_bitvector_expression(op, args)
        if op == '_':
            # Bit-vector constants
            return self.build_special_operator(expr)
        # Standard operations
        return self.build_standard_expression(op, args)

    def build_special_operator(self, expr):
        """Build special operators like bit-vector constants and extensions."""
        if expr[1].startswith('bv'):
            # Bit-vector constant
            value = int(expr[1][2:])  # Extract the value after 'bv'
            width = int(expr[2])  # Extract the bit-width
            return BitVecVal(value, width)
        if expr[1] == 'sign_extend':
            # Sign extension
            extension_bits = int(expr[2])
            value = self.build_expression(expr[3])
            return SignExt(extension_bits, value)
        if expr[1] == 'zero_extend':
            # Zero extension
            extension_bits = int(expr[2])
            value = self.build_expression(expr[3])
            return ZeroExt(extension_bits, value)
        if expr[1] == 'extract':
            # Bit extraction
            high = int(expr[2])
            low = int(expr[3])
            value = self.build_expression(expr[4])
            return Extract(high, low, value)
        raise ValueError(f"Unknown special operator: {expr[1]}")

    def build_let_expression(self, expr):
        """Build let expression with local bindings."""
        bindings = expr[1]
        body = expr[2]

        # Create a mapping of variable names to their expressions
        for binding in bindings:
            var_name = binding[0]
            var_expr = self.build_expression(binding[1])
            self.let_bindings[var_name] = var_expr

        return self.build_expression(body)

    def build_standard_expression(self, op, args):
        """Build expression for standard operations."""
        if op == '+':
            return sum(args)
        if op == '-':
            return args[0] - args[1] if len(args) == 2 else -args[0]
        if op == '*':
            return reduce(lambda x, y: x * y, args)
        if op == '/':
            return args[0] / args[1]
        if op == 'div':
            return args[0] / args[1]
        if op == '>':
            return args[0] > args[1]
        if op == '<':
            return args[0] < args[1]
        if op == '>=':
            return args[0] >= args[1]
        if op == '<=':
            return args[0] <= args[1]
        if op == '=':
            return args[0] == args[1]
        if op == 'mod':
            return args[0] % args[1]
        if op == 'distinct':
            return Distinct(*args)
        if op == 'concat':
            return Concat(*args)
        if op == 'and':
            return And(*args)
        if op == 'or':
            return Or(*args)
        if op == 'not':
            return Not(args[0])
        if op == '=>':
            return Implies(args[0], args[1])
        if op == 'ite':
            return If(args[0], args[1], args[2])
        raise ValueError(f"Unknown operator: {op}")

    def build_fp_expression(self, op, args):
        """Build floating-point expressions."""
        rm = RNE()  # Default rounding mode
        if op == 'fp.add':
            return fpAdd(rm, args[0], args[1])
        if op == 'fp.sub':
            return fpSub(rm, args[0], args[1])
        if op == 'fp.mul':
            return fpMul(rm, args[0], args[1])
        if op == 'fp.div':
            return fpDiv(rm, args[0], args[1])
        if op == 'fp.neg':
            return fpNeg(args[0])
        if op == 'fp.abs':
            return fpAbs(args[0])
        if op == 'fp.lt':
            return fpLT(args[0], args[1])
        if op == 'fp.gt':
            return fpGT(args[0], args[1])
        if op == 'fp.leq':
            return fpLEQ(args[0], args[1])
        if op == 'fp.geq':
            return fpGEQ(args[0], args[1])
        if op == 'fp.eq':
            return fpEQ(args[0], args[1])
        if op == 'fp.isNaN':
            return fpIsNaN(args[0])
        if op == 'fp.isInfinite':
            return fpIsInf(args[0])
        if op == 'fp.isZero':
            return fpIsZero(args[0])
        raise ValueError(f"Unknown FP operator: {op}")

    def build_bitvector_expression(self, op, args):
        """Build bit-vector expression with comprehensive operation support."""
        # Comparison operations
        if op == 'bvult':
            return ULT(args[0], args[1])
        if op == 'bvule':
            return ULE(args[0], args[1])
        if op == 'bvugt':
            return UGT(args[0], args[1])
        if op == 'bvuge':
            return UGE(args[0], args[1])
        if op == 'bvslt':
            return args[0] < args[1]
        if op == 'bvsle':
            return args[0] <= args[1]
        if op == 'bvsgt':
            return args[0] > args[1]
        if op == 'bvsge':
            return args[0] >= args[1]

        # Arithmetic operations
        if op == 'bvneg':
            return -args[0]
        if op == 'bvadd':
            return args[0] + args[1]
        if op == 'bvsub':
            return args[0] - args[1]
        if op == 'bvmul':
            return args[0] * args[1]
        if op == 'bvudiv':
            return UDiv(args[0], args[1])
        if op == 'bvsdiv':
            return args[0] / args[1]
        if op == 'bvurem':
            return URem(args[0], args[1])
        if op == 'bvsrem':
            return SRem(args[0], args[1])
        if op == 'bvsmod':
            # return SMod(args[0], args[1])
            return args[0] % args[1]

        # Bitwise operations
        if op == 'bvand':
            return args[0] & args[1]
        if op == 'bvor':
            return args[0] | args[1]
        if op == 'bvxor':
            return args[0] ^ args[1]
        if op == 'bvnot':
            return ~args[0]
        if op == 'bvnand':
            return ~(args[0] & args[1])
        if op == 'bvnor':
            return ~(args[0] | args[1])
        if op == 'bvxnor':
            return ~(args[0] ^ args[1])

        # Shift operations
        if op == 'bvshl':
            return args[0] << args[1]
        if op == 'bvlshr':
            return LShR(args[0], args[1])
        if op == 'bvashr':
            return args[0] >> args[1]

        raise ValueError(f"Unknown bit-vector operator: {op}")

    def parse_string(self, content):
        """Parse SMT-LIB content from string."""
        tokens = self.tokenize(content)
        while tokens:
            command = self.parse_tokens(tokens)
            if command:
                self.process_command(command)

    def parse_file(self, filename):
        """Parse SMT-LIB content from file."""
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        self.parse_string(content)

    def extract_scope_constraints(self):
        """Extract and print scope constraints."""
        print("precond:\n->", self.precond)
        print("cnt_list:")
        for cnt in self.cnt_list:
            print("->", cnt)

        return self.precond, self.cnt_list

    def get_precond(self):
        """Get precondition."""
        return self.precond

    def get_cnt_list(self):
        """Get constraint list."""
        return self.cnt_list
