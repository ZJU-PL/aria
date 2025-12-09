import aria.utils.ply.yacc as yacc
import aria.synthesis.spyro.lexer as lexer

tokens = lexer.tokens

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIV'),
    ('right', 'UMINUS')
)

def p_rulelist(p):
    '''rulelist : rule
                | rulelist rule'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[2])
    else:
        p[0] = [p[1]]

def p_rule(p):
    "rule : type symbol ARROW exprlist SEMI"

    p[0] = (p[1], p[2], p[4])

def p_type(p):
    "type : ID"

    p[0] = p[1]

def p_symbol(p):
    "symbol : ID"

    p[0] = p[1]

def p_exprlist(p):
    '''exprlist : expr
                | exprlist SPLITTER expr'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_expr_lambda(p):
    "expr : LPAREN ID RPAREN ARROW expr"

    p[0] = ('LAMBDA', p[2], p[5])

def p_expr_uminus(p):
    "expr : MINUS expr %prec UMINUS"

    p[0] = ('UNARY', '-', p[2])

def p_expr_unaryop(p):
    "expr : NOT expr"

    p[0] = ('UNARY', p[1], p[2])

def p_expr_binop(p):
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
    "expr : LPAREN expr RPAREN"

    p[0] = p[2]

def p_expr_var(p):
    "expr : ID"

    p[0] = ('VAR', p[1])

def p_expr_hole(p):
    '''expr : HOLE
            | HOLE LPAREN INT RPAREN'''

    if len(p) > 2:
        p[0] = ('HOLE', p[3])
    else:
        p[0] = ('HOLE', 0)

def p_expr_num(p):
    "expr : INT"

    p[0] = ('INT', p[1])

def p_expr_call(p):
    '''expr : ID LPAREN RPAREN
            | ID LPAREN args RPAREN'''
    if len(p) > 4:
        p[0] = ('FCALL', p[1], p[3])
    else:
        p[0] = ('FCALL', p[1], [])

def p_args(p):
    '''args : expr
            | args COMMA expr'''

    if len(p) > 2:
        p[0] = p[1]
        p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()
