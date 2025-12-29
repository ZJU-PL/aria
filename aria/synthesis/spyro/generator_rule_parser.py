"""Parser for generator rules in Spyro synthesis."""

from aria.utils.ply import yacc
from aria.synthesis.spyro import lexer

tokens = lexer.tokens

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIV'),
    ('right', 'UMINUS')
)


def p_rulelist(p):
    """Parse rule list."""
    '''rulelist : rule
                | rulelist rule'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[2])
    else:
        p[0] = [p[1]]


def p_rule(p):
    """Parse a rule."""
    "rule : type symbol ARROW exprlist SEMI"

    p[0] = (p[1], p[2], p[4])


def p_type(p):
    """Parse a type."""
    "type : ID"

    p[0] = p[1]


def p_symbol(p):
    """Parse a symbol."""
    "symbol : ID"

    p[0] = p[1]


def p_exprlist(p):
    """Parse expression list."""
    '''exprlist : expr
                | exprlist SPLITTER expr'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]


def p_expr_lambda(p):
    """Parse lambda expression."""
    "expr : LPAREN ID RPAREN ARROW expr"

    p[0] = ('LAMBDA', p[2], p[5])


def p_expr_uminus(p):
    """Parse unary minus expression."""
    "expr : MINUS expr %prec UMINUS"

    p[0] = ('UNARY', '-', p[2])


def p_expr_unaryop(p):
    """Parse unary operator expression."""
    "expr : NOT expr"

    p[0] = ('UNARY', p[1], p[2])


def p_expr_binop(p):
    """Parse binary operator expression."""
    '''expr : expr PLUS expr
            | expr MINUS expr
            | expr TIMES expr
            | expr DIV expr
            | expr LT expr
            | expr LE expr
            | expr GT expr
            | expr GE expr
            | expr EQ expr
            | expr NEQ expr
            | expr AND expr
            | expr OR expr'''

    p[0] = ('BINOP', p[2], p[1], p[3])


def p_expr_paren(p):
    """Parse parenthesized expression."""
    "expr : LPAREN expr RPAREN"

    p[0] = p[2]


def p_expr_var(p):
    """Parse variable expression."""
    "expr : ID"

    p[0] = ('VAR', p[1])


def p_expr_hole(p):
    """Parse hole expression."""
    '''expr : HOLE
            | HOLE LPAREN INT RPAREN'''

    if len(p) > 2:
        p[0] = ('HOLE', p[3])
    else:
        p[0] = ('HOLE', 0)


def p_expr_num(p):
    """Parse number expression."""
    "expr : INT"

    p[0] = ('INT', p[1])


def p_expr_call(p):
    """Parse function call expression."""
    '''expr : ID LPAREN RPAREN
            | ID LPAREN args RPAREN'''
    if len(p) > 4:
        p[0] = ('FCALL', p[1], p[3])
    else:
        p[0] = ('FCALL', p[1], [])


def p_args(p):
    """Parse function arguments."""
    '''args : expr
            | args COMMA expr'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]


def p_error(p):
    """Handle parser errors."""
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at EOF")


parser = yacc.yacc()
