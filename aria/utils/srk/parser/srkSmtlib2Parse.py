"""
PLY parser for SMT-LIB 2 format. Ported from srkSmtlib2Parse.mly (Menhir).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ply"))

import aria.utils.ply.yacc as yacc
from fractions import Fraction
from .srkSmtlib2Lex import tokens, make_lexer
from .srkSmtlib2Defs import (
    Constant, Symbol, Identifier, Index, Sort, QualId,
    Term, QuantifiedTerm, LetTerm, Model, FunctionDefinition,
    Attribute, AttributeValue, SExpr,
)


precedence = ()


def p_main_term(p):
    """main_term : term"""
    p[0] = p[1]


def p_main_model(p):
    """main_model : model"""
    p[0] = p[1]


# Model rules
def p_model(p):
    """model : LPAREN MODEL model_responses RPAREN
             | model_responses"""
    if len(p) == 5:
        functions = [fd for fd in p[3] if isinstance(fd, FunctionDefinition)]
        sorts = [fd for fd in p[3] if not isinstance(fd, FunctionDefinition)]
        p[0] = Model(functions=functions, sorts=sorts)
    else:
        functions = [fd for fd in p[1] if isinstance(fd, FunctionDefinition)]
        sorts = [fd for fd in p[1] if not isinstance(fd, FunctionDefinition)]
        p[0] = Model(functions=functions, sorts=sorts)


def p_model_responses_many(p):
    """model_responses : model_response model_responses"""
    p[0] = [p[1]] + p[2]


def p_model_responses_one(p):
    """model_responses : model_response"""
    p[0] = [p[1]]


def p_model_response(p):
    """model_response : LPAREN DEFFUN function_def RPAREN"""
    p[0] = p[3]


def p_function_def_simple(p):
    """function_def : SYMBOL LPAREN RPAREN sort term"""
    p[0] = FunctionDefinition(name=p[1], parameters=[], return_type=p[4], body=p[5])


def p_function_def_params(p):
    """function_def : SYMBOL LPAREN sorted_var_list RPAREN sort term"""
    p[0] = FunctionDefinition(name=p[1], parameters=p[3], return_type=p[5], body=p[6])


# Term rules
def p_term_spec_constant(p):
    """term : spec_constant"""
    qual_id = QualId(identifier=Identifier(symbol="const", indices=[]), sort=None)
    p[0] = Term(qual_id=qual_id, arguments=[Term(qual_id=QualId(identifier=Identifier(symbol=str(p[1]), indices=[]), sort=None), arguments=[])])


def p_term_qual_identifier(p):
    """term : qual_identifier"""
    p[0] = Term(qual_id=p[1], arguments=[])


def p_term_application(p):
    """term : LPAREN qual_identifier term_list RPAREN"""
    p[0] = Term(qual_id=p[2], arguments=p[3])


def p_term_let(p):
    """term : LPAREN LET LPAREN var_binding_list RPAREN term RPAREN"""
    p[0] = LetTerm(bindings=p[4], body=p[6])


def p_term_forall(p):
    """term : LPAREN FORALL LPAREN sorted_var_list RPAREN term RPAREN"""
    p[0] = QuantifiedTerm(quantifier="forall", variables=p[4], body=p[6])


def p_term_exists(p):
    """term : LPAREN EXISTS LPAREN sorted_var_list RPAREN term RPAREN"""
    p[0] = QuantifiedTerm(quantifier="exists", variables=p[4], body=p[6])


def p_term_annotated(p):
    """term : LPAREN BANG term attribute_list RPAREN"""
    p[0] = p[3]


def p_term_indexed(p):
    """term : LPAREN UNDERSCORE SYMBOL index_list RPAREN"""
    ident = Identifier(symbol=p[3], indices=p[4])
    p[0] = Term(qual_id=QualId(identifier=ident, sort=None), arguments=[])


def p_term_list_many(p):
    """term_list : term term_list"""
    p[0] = [p[1]] + p[2]


def p_term_list_one(p):
    """term_list : term"""
    p[0] = [p[1]]


def p_spec_constant_numeral(p):
    """spec_constant : NUMERAL"""
    p[0] = Constant(value=int(p[1]))


def p_spec_constant_decimal(p):
    """spec_constant : DECIMAL"""
    p[0] = Constant(value=p[1])


def p_spec_constant_string(p):
    """spec_constant : STRING"""
    p[0] = Constant(value=str(p[1]))


# Qualified identifier
def p_qual_identifier_simple(p):
    """qual_identifier : identifier"""
    p[0] = QualId(identifier=p[1], sort=None)


def p_qual_identifier_as(p):
    """qual_identifier : LPAREN AS identifier sort RPAREN"""
    p[0] = QualId(identifier=p[3], sort=p[4])


# Identifier
def p_identifier_simple(p):
    """identifier : SYMBOL"""
    p[0] = Identifier(symbol=p[1], indices=[])


def p_identifier_indexed(p):
    """identifier : LPAREN UNDERSCORE SYMBOL index_list RPAREN"""
    p[0] = Identifier(symbol=p[3], indices=p[4])


# Index
def p_index_numeral(p):
    """index : NUMERAL"""
    p[0] = Index(value=int(p[1]))


def p_index_symbol(p):
    """index : SYMBOL"""
    p[0] = Index(value=p[1])


def p_index_list_many(p):
    """index_list : index index_list"""
    p[0] = [p[1]] + p[2]


def p_index_list_one(p):
    """index_list : index"""
    p[0] = [p[1]]


# Sort
def p_sort_simple(p):
    """sort : identifier"""
    p[0] = Sort(identifier=p[1], arguments=[])


def p_sort_parametric(p):
    """sort : LPAREN identifier sort_list RPAREN"""
    p[0] = Sort(identifier=p[2], arguments=p[3])


def p_sort_list_many(p):
    """sort_list : sort sort_list"""
    p[0] = [p[1]] + p[2]


def p_sort_list_one(p):
    """sort_list : sort"""
    p[0] = [p[1]]


# Variable bindings (let)
def p_var_binding(p):
    """var_binding : LPAREN SYMBOL term RPAREN"""
    p[0] = (p[2], p[3])


def p_var_binding_list_many(p):
    """var_binding_list : var_binding var_binding_list"""
    p[0] = [p[1]] + p[2]


def p_var_binding_list_one(p):
    """var_binding_list : var_binding"""
    p[0] = [p[1]]


# Sorted variables (quantifiers)
def p_sorted_var(p):
    """sorted_var : LPAREN SYMBOL sort RPAREN"""
    p[0] = (p[2], p[3])


def p_sorted_var_list_many(p):
    """sorted_var_list : sorted_var sorted_var_list"""
    p[0] = [p[1]] + p[2]


def p_sorted_var_list_one(p):
    """sorted_var_list : sorted_var"""
    p[0] = [p[1]]


# Attributes
def p_attribute_keyword(p):
    """attribute : KEYWORD"""
    p[0] = Attribute(keyword=p[1], value=None)


def p_attribute_keyword_value(p):
    """attribute : LPAREN KEYWORD attribute_value RPAREN"""
    p[0] = Attribute(keyword=p[2], value=p[3])


def p_attribute_list_many(p):
    """attribute_list : attribute attribute_list"""
    p[0] = [p[1]] + p[2]


def p_attribute_list_one(p):
    """attribute_list : attribute"""
    p[0] = [p[1]]


def p_attribute_value_const(p):
    """attribute_value : spec_constant"""
    p[0] = AttributeValue(content=p[1])


def p_attribute_value_symbol(p):
    """attribute_value : SYMBOL"""
    p[0] = AttributeValue(content=p[1])


def p_attribute_value_sexpr(p):
    """attribute_value : LPAREN sexpr_list RPAREN"""
    p[0] = AttributeValue(content=SExpr(content=p[2]))


# S-expressions (for attribute values, sorts, etc.)
def p_sexpr_list_many(p):
    """sexpr_list : sexpr sexpr_list"""
    p[0] = [p[1]] + p[2]


def p_sexpr_list_one(p):
    """sexpr_list : sexpr"""
    p[0] = [p[1]]


def p_sexpr_const(p):
    """sexpr : spec_constant"""
    p[0] = SExpr(content=p[1])


def p_sexpr_symbol(p):
    """sexpr : SYMBOL"""
    p[0] = SExpr(content=p[1])


def p_sexpr_keyword(p):
    """sexpr : KEYWORD"""
    p[0] = SExpr(content=p[1])


def p_sexpr_listexpr(p):
    """sexpr : LPAREN sexpr_list RPAREN"""
    p[0] = SExpr(content=p[2])


def p_error(p):
    if p:
        print(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}")
    else:
        print("Syntax error at EOF")


def make_term_parser():
    return yacc.yacc(start="main_term")


def make_model_parser():
    return yacc.yacc(start="main_model")


class SMTLib2Parser:
    def __init__(self):
        self.lexer = make_lexer()
        self._term_parser = make_term_parser()
        self._model_parser = make_model_parser()

    def parse_term(self, input_string):
        return self._term_parser.parse(input_string, lexer=self.lexer)

    def parse_model(self, input_string):
        return self._model_parser.parse(input_string, lexer=self.lexer)
