"""Lexer for Spyro synthesis language."""

from aria.utils.ply import lex

tokens = (
    "SPLITTER",
    "ID",
    "INT",
    "LPAREN",
    "RPAREN",
    "ARROW",
    "COMMA",
    "HOLE",
    "SEMI",
    "LT",
    "LE",
    "GT",
    "GE",
    "AND",
    "OR",
    "NOT",
    "EQ",
    "NEQ",
    "PLUS",
    "MINUS",
    "TIMES",
    "DIV",
)

t_SPLITTER = r"\|"
t_ID = r"[A-Za-z_][A-Za-z0-9_]*"
t_INT = r"\d+"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_ARROW = r"->"
t_COMMA = r","
t_HOLE = r"\?\?"
t_SEMI = r";"
t_LT = r"<"
t_LE = r"<="
t_GT = r">"
t_GE = r">="
t_AND = r"&&"
t_OR = r"\|\|"
t_NOT = r"!"
t_EQ = r"=="
t_NEQ = r"!="
t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DIV = r"/"
t_ignore = " \t\r\n"


def t_error(t):
    """Handle lexer errors."""
    print(f"Illegal character {repr(t.value[0])}")
    t.lexer.skip(1)


lexer = lex.lex(debug=0)
