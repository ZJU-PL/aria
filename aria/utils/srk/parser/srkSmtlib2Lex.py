"""
PLY lexer for SMT-LIB 2 format. Ported from srkSmtlib2Lex.mll (ocamllex).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ply"))

import aria.utils.ply.lex as lex
from fractions import Fraction

tokens = (
    "SYMBOL",
    "KEYWORD",
    "STRING",
    "NUMERAL",
    "DECIMAL",
    "LPAREN",
    "RPAREN",
    "UNDERSCORE",
    "BANG",
)

reserved = {
    "as": "AS",
    "let": "LET",
    "forall": "FORALL",
    "exists": "EXISTS",
    "model": "MODEL",
    "define-fun": "DEFFUN",
    "match": "MATCH",
    "lambda": "LAMBDA",
    "declare-sort": "DECLARESORT",
    "define-sort": "DEFINESORT",
}

reserved_tokens = tuple(set(reserved.values()))
tokens = tokens + reserved_tokens

t_LPAREN = r"\("
t_RPAREN = r"\)"
t_UNDERSCORE = r"_"
t_BANG = r"!"
t_ignore = " \t\r"


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value[1:-1]
    return t


def t_DECIMAL(t):
    r"(?:0|[1-9][0-9]*)\.[0-9]+"
    t.value = Fraction(t.value)
    return t


def t_NUMERAL_hex(t):
    r"\#x[0-9a-fA-F]+"
    t.value = int(t.value[2:], 16)
    t.type = "NUMERAL"
    return t


def t_NUMERAL_bin(t):
    r"\#b[01]+"
    t.value = int(t.value[2:], 2)
    t.type = "NUMERAL"
    return t


def t_NUMERAL(t):
    r"0|[1-9][0-9]*"
    t.value = int(t.value)
    return t


def t_KEYWORD(t):
    r":[a-zA-Z~!@$%^&*_\-+=<>.?/][a-zA-Z0-9~!@$%^&*_\-+=<>.?/]*"
    t.value = t.value[1:]
    return t


def t_SYMBOL_quoted(t):
    r"\|[^|\\]*\|"
    t.value = t.value[1:-1]
    t.type = "SYMBOL"
    for kw, tok_type in reserved.items():
        if t.value == kw:
            t.type = tok_type
            break
    return t


def t_SYMBOL(t):
    r"[a-zA-Z~!@$%^&*_\-+=<>.?/][a-zA-Z0-9~!@$%^&*_\-+=<>.?/]*"
    for kw, tok_type in reserved.items():
        if t.value == kw:
            t.type = tok_type
            return t
    t.type = "SYMBOL"
    return t


def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)


def make_lexer():
    return lex.lex()


if __name__ == "__main__":
    lexer = make_lexer()
    test_input = '(model (define-fun x () Int 42) (define-fun y () Real 3.14))'
    lexer.input(test_input)
    for tok in lexer:
        print(f"  {tok.type}: {tok.value}")
