"""
SRK Parser using PLY (Python Lex-Yacc).

This module provides parsing for SRK formulas, equivalent to srkParse.mly.
It parses mathematical expressions and formulas in the SRK syntax.
"""

from __future__ import annotations
import sys
import os
from typing import List, Tuple, Optional, Any

# Add ply directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ply'))

import aria.utils.ply.yacc as yacc

# Import the lexer
from aria.srk.srkLex import tokens, make_lexer
from aria.srk.srkAst import SimplifyingContext
from aria.srk.syntax import Context, Symbol, Expression, FormulaExpression, ArithExpression, Type, Const
from aria.srk.syntax import mk_symbol, mk_const, mk_real, mk_add, mk_mul, mk_neg, mk_div
from aria.srk.syntax import mk_eq, mk_leq, mk_lt, mk_geq, mk_and, mk_or, mk_not, mk_ite
from aria.srk.syntax import mk_forall, mk_exists
from aria.srk.qQ import QQ

# Global context for parsing
Ctx = SimplifyingContext()

# Symbol memo table
_symbol_memo = {}

def symbol_of_string(name: str) -> Symbol:
    """Create or retrieve a symbol with the given name."""
    if name not in _symbol_memo:
        _symbol_memo[name] = mk_symbol(Ctx.context, name, Type.REAL)
    return _symbol_memo[name]

# Helper functions for missing operations
def mk_gt(left: ArithExpression, right: ArithExpression) -> FormulaExpression:
    """Create a greater-than comparison (a > b is equivalent to b < a)."""
    return mk_lt(right, left)

def mk_pow(base: ArithExpression, exponent: ArithExpression) -> ArithExpression:
    """Create a power expression (base^exponent)."""
    # For now, implement as multiplication for simple cases
    # A full implementation would need proper power handling
    if isinstance(exponent, Const) and hasattr(exponent, 'value'):
        if exponent.value == 2:
            return mk_mul([base, base])
        elif exponent.value == 1:
            return base
        elif exponent.value == 0:
            return mk_real(1.0)
    # For complex cases, use multiplication as approximation
    return mk_mul([base, base])  # Simplified implementation


# Precedence and associativity (from lowest to highest)
precedence = (
    ('right', 'ITE'),           # if-then-else (right associative)
    ('left', 'OR'),             # or
    ('left', 'AND'),            # and
    ('left', 'EQ', 'NEQ'),      # =, !=
    ('left', 'LT', 'LEQ', 'GT', 'GEQ'),  # <, <=, >, >=
    ('left', 'ADD', 'MINUS'),   # +, -
    ('left', 'MUL', 'DIV'),     # *, /
    ('right', 'POW'),           # ^ (exponentiation, right associative)
    ('right', 'UMINUS'), # unary -
)


# Grammar rules

# Main entry point
def p_expression(p):
    """expression : formula"""
    p[0] = p[1]

# Formula rules (logical expressions)
def p_formula(p):
    """formula : formula_or"""
    p[0] = p[1]

def p_formula_or(p):
    """formula_or : formula_or OR formula_and
                  | formula_and"""
    if len(p) == 4:
        p[0] = mk_or([p[1], p[3]])
    else:
        p[0] = p[1]

def p_formula_and(p):
    """formula_and : formula_and AND formula_comp
                   | formula_comp"""
    if len(p) == 4:
        p[0] = mk_and([p[1], p[3]])
    else:
        p[0] = p[1]

def p_formula_comp(p):
    """formula_comp : term EQ term
                    | term NEQ term
                    | term LT term
                    | term LEQ term
                    | term GT term
                    | term GEQ term
                    | formula_not"""
    if len(p) == 4:
        if p[2] == '=':
            p[0] = mk_eq(p[1], p[3])
        elif p[2] == '!=':
            p[0] = mk_not(mk_eq(p[1], p[3]))
        elif p[2] == '<':
            p[0] = mk_lt(p[1], p[3])
        elif p[2] == '<=':
            p[0] = mk_leq(p[1], p[3])
        elif p[2] == '>':
            p[0] = mk_gt(p[1], p[3])
        elif p[2] == '>=':
            p[0] = mk_geq(p[1], p[3])
    else:
        p[0] = p[1]

def p_formula_not(p):
    """formula_not : NOT formula_not
                   | formula_ite
                   | formula_quant
                   | LPAREN formula RPAREN"""
    if len(p) == 3:
        p[0] = mk_not(p[2])
    elif len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = p[1]

def p_formula_ite(p):
    """formula_ite : ITE formula THEN formula ELSE formula
                   | formula_quant"""
    if len(p) == 6:
        p[0] = mk_ite(p[2], p[4], p[6])
    else:
        p[0] = p[1]

def p_formula_quant(p):
    """formula_quant : FORALL ID DOT formula
                     | EXISTS ID DOT formula
                     | term"""
    if len(p) == 5:
        if p[1] == 'forall':
            p[0] = mk_forall(p[2], Type.REAL, p[4])
        elif p[1] == 'exists':
            p[0] = mk_exists(p[2], Type.REAL, p[4])
    else:
        p[0] = p[1]




# Term rules (arithmetic expressions)
def p_term(p):
    """term : term_add"""
    p[0] = p[1]

def p_term_add(p):
    """term_add : term_add ADD term_mul
                | term_add MINUS term_mul
                | term_mul"""
    if len(p) == 4:
        if p[2] == '+':
            p[0] = mk_add([p[1], p[3]])
        elif p[2] == '-':
            p[0] = mk_add([p[1], mk_neg(p[3])])
    else:
        p[0] = p[1]

def p_term_mul(p):
    """term_mul : term_mul MUL term_pow
                | term_mul DIV term_pow
                | term_pow"""
    if len(p) == 4:
        if p[2] == '*':
            p[0] = mk_mul([p[1], p[3]])
        elif p[2] == '/':
            p[0] = mk_div(p[1], p[3])
    else:
        p[0] = p[1]

def p_term_pow(p):
    """term_pow : term_pow POW term_unary
                | term_unary"""
    if len(p) == 4:
        p[0] = mk_pow(p[1], p[3])
    else:
        p[0] = p[1]

def p_term_unary(p):
    """term_unary : MINUS term_unary %prec UMINUS
                  | term_primary"""
    if len(p) == 3:
        p[0] = mk_neg(p[2])
    else:
        p[0] = p[1]

def p_term_primary(p):
    """term_primary : ID
                    | REAL
                    | LPAREN term RPAREN"""
    if len(p) == 2:
        if isinstance(p[1], str) and p[1].isalpha():
            p[0] = mk_const(symbol_of_string(p[1]))
        else:
            p[0] = mk_real(p[1])
    else:
        p[0] = p[2]




# SMT2 Formula rules (simplified)
def p_smt2_formula(p):
    """smt2_formula : LPAREN up_smt2_formula RPAREN"""
    p[0] = p[2]

def p_up_smt2_formula_and(p):
    """up_smt2_formula : AND smt2_formula_list"""
    p[0] = mk_and(p[2])

def p_up_smt2_formula_or(p):
    """up_smt2_formula : OR smt2_formula_list"""
    p[0] = mk_or(p[2])

def p_up_smt2_formula_gt(p):
    """up_smt2_formula : GT smt2_term smt2_term"""
    p[0] = mk_gt(p[2], p[3])

def p_smt2_formula_list(p):
    """smt2_formula_list : smt2_formula
                        | smt2_formula_list smt2_formula"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

# SMT2 Term rules (simplified)
def p_smt2_term(p):
    """smt2_term : LPAREN up_smt2_term RPAREN"""
    p[0] = p[2]

def p_up_smt2_term_add(p):
    """up_smt2_term : ADD smt2_term_list"""
    p[0] = mk_add(p[2])

def p_up_smt2_term_mul(p):
    """up_smt2_term : MUL smt2_term_list"""
    p[0] = mk_mul(p[2])

def p_up_smt2_term_neg(p):
    """up_smt2_term : MINUS smt2_term"""
    p[0] = mk_neg(p[2])

def p_up_smt2_term_real(p):
    """up_smt2_term : REAL"""
    p[0] = mk_real(p[1])

def p_smt2_term_list(p):
    """smt2_term_list : smt2_term
                     | smt2_term_list smt2_term"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


# Error handling
def p_error(p):
    if p:
        print(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}")
    else:
        print("Syntax error at EOF")


# Build the parser
def make_parser():
    """Create and return a parser for SRK formulas."""
    return yacc.yacc(start='expression')

def make_smt2_parser():
    """Create and return a parser for SMT2 formulas."""
    return yacc.yacc(start='smt2_formula')

class MathParser:
    """Parser for math formulas."""

    def __init__(self):
        self.lexer = make_lexer()
        self.parser = make_parser()
        self.context = Ctx

    def parse(self, input_string: str) -> Any:
        """Parse an input string and return the AST."""
        global Ctx
        # Reset context for fresh parse
        Ctx = SimplifyingContext()
        _symbol_memo.clear()

        result = self.parser.parse(input_string, lexer=self.lexer)
        return result

class SMT2Parser:
    """Parser for SMT2 formulas."""

    def __init__(self):
        self.lexer = make_lexer()
        self.parser = make_smt2_parser()
        self.context = Ctx

    def parse(self, input_string: str) -> Any:
        """Parse an input string and return the AST."""
        global Ctx
        # Reset context for fresh parse
        Ctx = SimplifyingContext()
        _symbol_memo.clear()

        result = self.parser.parse(input_string, lexer=self.lexer)
        return result

# Convenience functions
def parse_formula(input_string: str) -> FormulaExpression:
    """Parse a math formula string and return the AST."""
    parser = MathParser()
    return parser.parse(input_string)

def parse_smt2(input_string: str) -> FormulaExpression:
    """Parse an SMT2 formula string and return the AST."""
    parser = SMT2Parser()
    return parser.parse(input_string)


# Testing
if __name__ == '__main__':
    # Test the parser
    test_inputs = [
        "x + y <= 10",
        "x * y = 5",
        "ForAll[{x, y}, And[x + y <= 10, x * y = 5]]",
        "Exists[{z}, z < x]",
    ]

    print("Testing SRK Parser:")
    parser = MathParser()

    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        try:
            result = parser.parse_formula(test_input)
            print(f"  Parsed: {result}")
        except Exception as e:
            print(f"  Error: {e}")


# PLY availability flag
PLY_AVAILABLE = True
