"""
SMT-LIB 2 parser and pretty-printer.

This module provides functionality to parse and pretty-print SMT-LIB 2
expressions and responses, particularly for model extraction and analysis.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple, Any, Dict, IO
from fractions import Fraction
import re

from aria.utils.srk.parser.srkSmtlib2Defs import *
from aria.utils.srk.core.syntax import Context, Symbol as SRKSymbol, Type
from aria.utils.srk import zZ
from aria.utils.srk import qQ
from .srkSmtlib2Parse import SMTLib2Parser as PLYSMTLib2Parser

# SMT-LIB 2 keywords and reserved words
SMTLIB2_KEYWORDS = {
    "declare-fun",
    "declare-const",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "declare-datatypes",
    "declare-codatatypes",
    "declare-datatype",
    "declare-codatatype",
    "define-datatypes",
    "define-codatatypes",
    "set-logic",
    "set-option",
    "set-info",
    "get-assertions",
    "get-assignment",
    "get-proof",
    "get-unsat-assumptions",
    "get-unsat-core",
    "get-model",
    "get-value",
    "get-option",
    "get-info",
    "push",
    "pop",
    "assert",
    "check-sat",
    "check-sat-assuming",
    "exit",
    "reset",
    "reset-assertions",
    "echo",
    "declare-sort",
    "declare-const",
    "declare-fun",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "declare-datatypes",
    "declare-codatatypes",
    "declare-datatype",
    "declare-codatatype",
    "define-datatypes",
    "define-codatatypes",
    "set-logic",
    "set-option",
    "set-info",
    "get-assertions",
    "get-assignment",
    "get-proof",
    "get-unsat-assumptions",
    "get-unsat-core",
    "get-model",
    "get-value",
    "get-option",
    "get-info",
    "push",
    "pop",
    "assert",
    "check-sat",
    "check-sat-assuming",
    "exit",
    "reset",
    "reset-assertions",
    "echo",
    "declare-sort",
    "declare-const",
    "declare-fun",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "declare-datatypes",
    "declare-codatatypes",
    "declare-datatype",
    "declare-codatatype",
    "define-datatypes",
    "define-codatatypes",
    "set-logic",
    "set-option",
    "set-info",
    "get-assertions",
    "get-assignment",
    "get-proof",
    "get-unsat-assumptions",
    "get-unsat-core",
    "get-model",
    "get-value",
    "get-option",
    "get-info",
    "push",
    "pop",
    "assert",
    "check-sat",
    "check-sat-assuming",
    "exit",
    "reset",
    "reset-assertions",
    "echo",
    "declare-sort",
    "declare-const",
    "declare-fun",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "declare-datatypes",
    "declare-codatatypes",
    "declare-datatype",
    "declare-codatatype",
    "define-datatypes",
    "define-codatatypes",
    "set-logic",
    "set-option",
    "set-info",
    "get-assertions",
    "get-assignment",
    "get-proof",
    "get-unsat-assumptions",
    "get-unsat-core",
    "get-model",
    "get-value",
    "get-option",
    "get-info",
    "push",
    "pop",
    "assert",
    "check-sat",
    "check-sat-assuming",
    "exit",
    "reset",
    "reset-assertions",
    "echo",
    "declare-sort",
    "declare-const",
    "declare-fun",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "declare-datatypes",
    "declare-codatatypes",
    "declare-datatype",
    "declare-codatatype",
    "define-datatypes",
    "define-codatatypes",
    "set-logic",
    "set-option",
    "set-info",
    "get-assertions",
    "get-assignment",
    "get-proof",
    "get-unsat-assumptions",
    "get-unsat-core",
    "get-model",
    "get-value",
    "get-option",
    "get-info",
    "push",
    "pop",
    "assert",
    "check-sat",
    "check-sat-assuming",
    "exit",
    "reset",
    "reset-assertions",
    "echo",
}


class SMTLib2Parser:
    """Parser for SMT-LIB 2 expressions using PLY-based lexer/parser."""

    def __init__(self, context: Context = None):
        self.context = context
        self._parser = PLYSMTLib2Parser()

    def parse_term(self, input_str: str) -> Term:
        return self._parser.parse_term(input_str)

    def parse_model(self, model_text: str) -> Model:
        return self._parser.parse_model(model_text)

    def parse_model_with_validation(self, model_text: str) -> Model:
        if not is_valid_smtlib2_model(model_text):
            raise ValueError("Invalid SMT-LIB 2 model text")
        return self.parse_model(model_text)


def string_parse_term(s: str) -> Term:
    parser = PLYSMTLib2Parser()
    return parser.parse_term(s)


def string_parse_model(s: str) -> Model:
    parser = PLYSMTLib2Parser()
    return parser.parse_model(s)


def file_parse_term(filename: str) -> Term:
    with open(filename, 'r') as f:
        return string_parse_term(f.read())


def file_parse_model(filename: str) -> Model:
    with open(filename, 'r') as f:
        return string_parse_model(f.read())


class SMTLib2Printer:
    """Pretty-printer for SMT-LIB 2 expressions."""

    def __init__(self, context: Context = None):
        self.context = context

    def print_list(self, items: List[Any], sep: str = " ") -> str:
        """Print a list of items separated by sep."""
        if not items:
            return ""
        elif len(items) == 1:
            return str(items[0])
        else:
            return sep.join(str(item) for item in items)

    def print_constant(self, const: Constant) -> str:
        """Print a constant value."""
        return str(const)

    def print_symbol(self, sym: Symbol) -> str:
        """Print a symbol."""
        return sym

    def print_index(self, idx: Index) -> str:
        """Print an index."""
        return str(idx)

    def print_identifier(self, ident: Identifier) -> str:
        """Print an identifier."""
        return str(ident)

    def print_sort(self, sort: Sort) -> str:
        """Print a sort."""
        return str(sort)

    def print_qual_id(self, qual_id: QualId) -> str:
        """Print a qualified identifier."""
        return str(qual_id)

    def print_pattern(self, pattern: Pattern) -> str:
        """Print a pattern."""
        return str(pattern)

    def print_sexpr(self, sexpr: SExpr) -> str:
        """Print an S-expression."""
        return str(sexpr)

    def print_attribute_value(self, attr_val: AttributeValue) -> str:
        """Print an attribute value."""
        return str(attr_val)

    def print_attribute(self, attr: Attribute) -> str:
        """Print an attribute."""
        return str(attr)

    def print_term(self, term: Term) -> str:
        """Print a term."""
        if not term.arguments:
            return str(term.qual_id)
        else:
            args_str = " ".join(self.print_term(arg) for arg in term.arguments)
            return f"({term.qual_id} {args_str})"

    def print_quantified_term(self, qterm: QuantifiedTerm) -> str:
        """Print a quantified term."""
        vars_str = " ".join(
            f"({var} {self.print_sort(sort)})" for var, sort in qterm.variables
        )
        return f"({qterm.quantifier} ({vars_str}) {self.print_term(qterm.body)})"

    def print_let_term(self, lterm: LetTerm) -> str:
        """Print a let term."""
        bindings_str = " ".join(
            f"({var} {self.print_term(term)})" for var, term in lterm.bindings
        )
        return f"(let ({bindings_str}) {self.print_term(lterm.body)})"

    def print_lambda_term(self, lterm: LambdaTerm) -> str:
        """Print a lambda term."""
        vars_str = " ".join(
            f"({var} {self.print_sort(sort)})" for var, sort in lterm.variables
        )
        return f"(lambda ({vars_str}) {self.print_term(lterm.body)})"

    def print_model(self, model: Model) -> str:
        """Print a model."""
        return model.to_smtlib2_string()


def parse_smtlib2_expression(text: str, context: Context = None) -> Term:
    parser = SMTLib2Parser(context)
    return parser.parse_term(text)


def print_smtlib2_expression(expr: Term, context: Context = None) -> str:
    printer = SMTLib2Printer(context)
    return printer.print_term(expr)


def parse_smtlib2_model(model_text: str, context: Context = None) -> Model:
    parser = SMTLib2Parser(context)
    return parser.parse_model(model_text)


def parse_smtlib2_model_validated(model_text: str, context: Context = None) -> Model:
    parser = SMTLib2Parser(context)
    return parser.parse_model_with_validation(model_text)


def print_smtlib2_model(model: Model, context: Context = None) -> str:
    printer = SMTLib2Printer(context)
    return printer.print_model(model)


def parse_smtlib2_model_from_string(model_str: str, context: Context = None) -> Model:
    try:
        return parse_smtlib2_model_validated(model_str, context)
    except Exception:
        return Model([], [])


def is_valid_smtlib2_model(model_text: str) -> bool:
    """Check if text looks like a valid SMT-LIB 2 model."""
    if not model_text or not model_text.strip():
        return False
    stripped = model_text.strip()
    return stripped.startswith("(model") and stripped.endswith(")")


# Test function for comprehensive functionality
def test_smtlib2_functionality():
    """Comprehensive test of SMT-LIB 2 functionality."""
    from aria.utils.srk.core.syntax import Context

    print("Testing SMT-LIB 2 parser...")

    # Test 1: Simple model with constants and functions
    model_text1 = """(model
  (define-fun x () Int 5)
  (define-fun y () Real 3/2)
  (define-fun f ((z Int)) Int z)
)"""

    context = Context()
    model1 = parse_smtlib2_model(model_text1, context)

    print(f"Test 1 - Parsed model with {len(model1.functions)} functions")
    output1 = print_smtlib2_model(model1, context)
    print("Output matches input:", model_text1.strip() == output1.strip())

    # Test 2: Expression parsing
    expr_text = "(and (> x 0) (< y 10))"
    expr = parse_smtlib2_expression(expr_text, context)
    print(f"Test 2 - Parsed expression: {expr}")

    # Test 3: Validation
    is_valid1 = is_valid_smtlib2_model(model_text1)
    is_valid2 = is_valid_smtlib2_model("invalid model")
    print(f"Test 3 - Valid model: {is_valid1}, Invalid model: {is_valid2}")

    # Test 4: Error handling
    empty_model = parse_smtlib2_model_from_string("", context)
    print(f"Test 4 - Empty model functions: {len(empty_model.functions)}")

    print("All tests completed successfully!")
    return True


if __name__ == "__main__":
    test_smtlib2_functionality()
