import ply.lex as lex
import ply.yacc as yacc
import sympy as sp

from quantifier import Exists, ForAll

tokens = (
    'FORALL', 'EXISTS',
    'AND', 'OR',
    'LPAREN', 'RPAREN', 'LSQUARE', 'RSQUARE', 'LBRAKET', 'RBRAKET',
    'LEQ', 'LT', 'VARIABLE', 'NUMBER', 'COMMA',
    'PLUS', 'MINUS', 'TIMES'
)

t_FORALL = r'ForAll'
t_EXISTS = r'Exists'
t_AND = r'And'
t_OR = r'Or'
t_LEQ = r'<='
t_LT = r'<'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LSQUARE = r'\['
t_RSQUARE = r'\]'
t_LBRAKET = r'\{'
t_RBRAKET = r'\}'
t_COMMA = r','
t_ignore = ' \t\n'


def t_VARIABLE(t):
    r'v\d+'
    t.value = sp.Symbol(t.value)
    return t


def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)


lexer = lex.lex()


def p_expression_quantifier(p):
    '''expression : quantifier
                  | logical_expression'''
    p[0] = p[1]


def p_quantifier(p):
    '''quantifier : FORALL LSQUARE variable_list COMMA expression RSQUARE
                  | EXISTS LSQUARE variable_list COMMA expression RSQUARE'''
    if p[1] == 'ForAll':
        p[0] = ForAll(p[3], p[5])
    else:
        p[0] = Exists(p[3], p[5])


def p_variable_list(p):
    '''variable_list : VARIABLE
                     | LBRAKET variable_list RBRAKET
                     | variable_list COMMA VARIABLE'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 4 and p[1] == '{' and p[3] == '}':
        p[0] = p[2]
    else:
        p[0] = p[1] + [p[3]]


def p_logical_expression(p):
    '''logical_expression : logical_term
                          | AND LSQUARE logical_expression_list RSQUARE
                          | OR LSQUARE logical_expression_list RSQUARE'''
    if len(p) == 2:
        p[0] = p[1]
    elif p[1] == 'And':
        p[0] = sp.And(*p[3])
    else:
        p[0] = sp.Or(*p[3])


def p_logical_expression_list(p):
    '''logical_expression_list : logical_expression
                               | logical_expression_list COMMA logical_expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]


def p_logical_term(p):
    '''logical_term : LPAREN logical_term RPAREN
                    | arithmetic_expression LEQ arithmetic_expression
                    | arithmetic_expression LT arithmetic_expression'''
    if p[1] == '(' and p[3] == ')':
        p[0] = p[2]
    elif p[2] == '<=':
        p[0] = p[1] <= p[3]
    elif p[2] == '<':
        p[0] = p[1] < p[3]


def p_arithmetic_expression(p):
    '''arithmetic_expression : arithmetic_term
                             | arithmetic_expression PLUS arithmetic_term
                             | arithmetic_expression MINUS arithmetic_term'''
    if len(p) == 2:
        p[0] = p[1]
    elif p[2] == '+':
        p[0] = p[1] + p[3]
    else:
        p[0] = p[1] - p[3]


def p_arithmetic_term(p):
    '''arithmetic_term : arithmetic_term TIMES arithmetic_factor
                       | arithmetic_factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1] * p[3]


def p_arithmetic_factor(p):
    '''arithmetic_factor : arithmetic_value
                         | LPAREN arithmetic_expression RPAREN'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_arithmetic_value(p):
    '''arithmetic_value : value
                        | MINUS value'''
    if p[1] == '-':
        p[0] = -p[2]
    else:
        p[0] = p[1]


def p_value(p):
    '''value : NUMBER
             | VARIABLE'''
    p[0] = p[1]


def p_error(p):
    print("Syntax error at '%s'" % p.value if p else "Syntax error at EOF")


def parse_file(file_path):
    """
    Parse the formula with the .m extension from the given file.
    Using lex and yacc.

    Parameters
    ----------
    file_path : str
        The path to the file to be parsed.

    Returns
    -------
    List[Formula]
        The parsed formula.
    """
    with open(file_path, 'r') as f:
        data = f.read()
    parser = yacc.yacc()
    return [parser.parse(data)]
