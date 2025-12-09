"""
SRK Lexer using PLY (Python Lex-Yacc).

This module provides lexical analysis for SRK formulas, equivalent to srkLex.mll.
It tokenizes mathematical expressions and formulas in the SRK syntax.
"""

from __future__ import annotations
import sys
import os

# Add ply directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ply'))

import aria.utils.ply.lex as lex
from fractions import Fraction

# Token list
tokens = (
    'ID',
    'REAL',
    'OBJECTIVE',
    'AND',
    'OR',
    'NOT',
    'FORALL',
    'EXISTS',
    'LEQ',
    'GEQ',
    'EQ',
    'NEQ',
    'LT',
    'GT',
    'MUL',
    'DIV',
    'POW',
    'ADD',
    'MINUS',
    'ITE',
    'THEN',
    'ELSE',
    'DOT',
    'COMMA',
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'LBRACE',
    'RBRACE',
)

# Token rules
t_OBJECTIVE = r'Objective:'
t_AND = r'And'
t_OR = r'Or'
t_NOT = r'Not'
t_FORALL = r'ForAll'
t_EXISTS = r'Exists'
t_LEQ = r'<='
t_GEQ = r'>='
t_EQ = r'='
t_NEQ = r'!='
t_LT = r'<'
t_GT = r'>'
t_MUL = r'\*'
t_DIV = r'/'
t_POW = r'\^'
t_ADD = r'\+'
t_MINUS = r'-'
# t_UPLUS = r'\+'  # Commented out - conflicts with ADD
t_ITE = r'if'
t_THEN = r'then'
t_ELSE = r'else'
t_DOT = r'\.'
t_COMMA = r','
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LBRACE = r'\{'
t_RBRACE = r'\}'

# Ignored characters (whitespace and newlines)
t_ignore = ' \t'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Identifier token
def t_ID(t):
    r'[_a-zA-Z$?][_a-zA-Z0-9]*'
    # Check for reserved words
    reserved = {
        'Objective:': 'OBJECTIVE',
        'And': 'AND',
        'Or': 'OR',
        'Not': 'NOT',
        'ForAll': 'FORALL',
        'Exists': 'EXISTS',
        'and': 'AND',
        'or': 'OR',
        'forall': 'FORALL',
        'exists': 'EXISTS',
        'if': 'ITE',
        'then': 'THEN',
        'else': 'ELSE',
    }
    t.type = reserved.get(t.value, 'ID')
    return t

# Real number token (supports fractions like -5/3)
def t_REAL(t):
    r'-?[0-9]+(/[0-9]+)?'
    if '/' in t.value:
        parts = t.value.split('/')
        numerator = int(parts[0])
        denominator = int(parts[1])
        t.value = Fraction(numerator, denominator)
    else:
        t.value = Fraction(int(t.value))
    return t

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)

# Build the lexer
def make_lexer():
    """Create and return a lexer for SRK formulas."""
    return lex.lex()


class MathLexer:
    """Wrapper class for the math lexer."""

    def __init__(self):
        self.lexer = make_lexer()
        self.lineno = 1

    def input(self, data):
        """Set input data for the lexer."""
        self.lexer.input(data)

    def token(self):
        """Get the next token."""
        tok = self.lexer.token()
        if tok:
            self.lineno = tok.lineno
        return tok

    def __iter__(self):
        """Allow iteration over tokens."""
        return iter(self.lexer)


class SMT2Lexer:
    """Wrapper class for the SMT2 lexer (similar syntax but lowercase keywords)."""

    def __init__(self):
        self.lexer = make_lexer()
        self.lineno = 1

    def input(self, data):
        """Set input data for the lexer."""
        # Preprocess to lowercase certain keywords for SMT2 compatibility
        self.lexer.input(data)

    def token(self):
        """Get the next token."""
        tok = self.lexer.token()
        if tok:
            self.lineno = tok.lineno
        return tok


# Convenience functions
def tokenize(input_string: str) -> list:
    """Tokenize an input string and return a list of tokens."""
    lexer = make_lexer()
    lexer.input(input_string)

    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens.append(tok)

    return tokens


def tokenize_math(input_string: str) -> list:
    """Tokenize a math formula and return a list of tokens."""
    return tokenize(input_string)


def tokenize_smt2(input_string: str) -> list:
    """Tokenize an SMT2 formula and return a list of tokens."""
    return tokenize(input_string)


# Testing
if __name__ == '__main__':
    # Test the lexer
    test_input = """
    ForAll[{x, y}, And[x + y <= 10, x * y = 5]]
    Objective: x + y
    x = 3/2
    """

    print("Testing SRK Lexer:")
    print("Input:", test_input)
    print("\nTokens:")

    tokens = tokenize(test_input)
    for tok in tokens:
        print(f"  {tok.type}: {tok.value}")
