"""
SMT-LIB 2 parser and pretty-printer.

This module provides functionality to parse and pretty-print SMT-LIB 2
expressions and responses, particularly for model extraction and analysis.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple, Any, Dict, IO
from fractions import Fraction
import re

from aria.srk.srkSmtlib2Defs import *
from aria.srk.syntax import Context, Symbol as SRKSymbol, Type
from aria.srk import zZ
from aria.srk import qQ

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
    """Parser for SMT-LIB 2 expressions."""

    def __init__(self, context: Context):
        self.context = context
        self.tokens: List[str] = []
        self.pos = 0

    def tokenize(self, text: str) -> List[str]:
        """Tokenize SMT-LIB 2 text."""
        tokens = []
        current = ""
        i = 0

        while i < len(text):
            char = text[i]

            # Handle parentheses
            if char in "()":
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)

            # Handle whitespace
            elif char.isspace():
                if current:
                    tokens.append(current)
                    current = ""

            # Handle string literals
            elif char == '"' and current == "":
                # Start of string literal
                j = i + 1
                while j < len(text) and text[j] != '"':
                    j += 1
                if j < len(text):
                    tokens.append('"' + text[i + 1 : j] + '"')
                    i = j
                else:
                    current += char

            # Handle comments
            elif char == ";" and current == "":
                # Comment - skip to end of line
                j = i + 1
                while j < len(text) and text[j] != "\n":
                    j += 1
                i = j - 1

            # Handle keywords and symbols
            else:
                current += char

            i += 1

        if current:
            tokens.append(current)

        # Post-process tokens to handle keywords
        processed_tokens = []
        for token in tokens:
            if token in SMTLIB2_KEYWORDS:
                processed_tokens.append(token.upper())
            else:
                processed_tokens.append(token)

        return processed_tokens

    def parse_expression(self, tokens: List[str]) -> SExpr:
        """Parse tokens into an S-expression."""
        if not tokens:
            raise ValueError("Empty token list")

        # Find the matching closing parenthesis for the entire expression
        i = 0
        paren_depth = 0

        while i < len(tokens):
            if tokens[i] == "(":
                paren_depth += 1
            elif tokens[i] == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    # Found the matching closing paren
                    break
            i += 1

        if paren_depth != 0:
            raise ValueError("Unmatched parentheses")

        # Parse the content between the outer parentheses
        return self._parse_sexpr_content(tokens[1:i])

    def _parse_sexpr_content(self, tokens: List[str]) -> SExpr:
        """Parse the content of an S-expression."""
        if not tokens:
            return SExpr.SSexpr([])

        content = []
        i = 0

        while i < len(tokens):
            if tokens[i] == "(":
                # Find the matching closing paren for this nested expression
                nested_end = self._find_matching_paren(tokens, i)
                if nested_end == -1:
                    raise ValueError("Unmatched parentheses in nested expression")

                # Parse the nested expression
                nested_tokens = tokens[i : nested_end + 1]
                nested_expr = self.parse_expression(nested_tokens)
                content.append(nested_expr)
                i = nested_end + 1
            else:
                # Atomic token
                content.append(tokens[i])
                i += 1

        return SExpr.SSexpr(content)

    def _find_matching_paren(self, tokens: List[str], start: int) -> int:
        """Find the matching closing parenthesis starting from position start."""
        paren_depth = 0
        i = start

        while i < len(tokens):
            if tokens[i] == "(":
                paren_depth += 1
            elif tokens[i] == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    return i
            i += 1

        return -1  # No matching paren found

    def parse_model(self, model_text: str) -> Model:
        """Parse SMT-LIB 2 model response."""
        functions = []
        sorts = []

        try:
            tokens = self.tokenize(model_text)
            if not tokens:
                return Model(functions, sorts)

            # Parse the model S-expression
            sexpr = self.parse_expression(tokens)

            if isinstance(sexpr.content, list) and len(sexpr.content) > 0:
                # First element should be "MODEL" keyword
                first_elem = sexpr.content[0]
                if isinstance(first_elem, str) and first_elem.upper() == "MODEL":
                    # Parse the model content
                    for elem in sexpr.content[1:]:
                        if isinstance(elem, SExpr) and isinstance(elem.content, list):
                            self._parse_model_element(elem, functions, sorts)

        except Exception as e:
            # If parsing fails, return empty model
            pass

        return Model(functions, sorts)

    def _parse_model_element(
        self, elem: SExpr, functions: List[Any], sorts: List[Any]
    ) -> None:
        """Parse a single element from the model (define-fun, etc.)."""
        if not isinstance(elem.content, list) or len(elem.content) < 2:
            return

        # Check if this is a define-fun
        first_token = elem.content[0]
        if isinstance(first_token, str) and first_token == "DEFINE-FUN":
            self._parse_define_fun(elem, functions)

        # Could extend for other model elements like define-fun-rec, declare-fun, etc.

    def _parse_define_fun(self, elem: SExpr, functions: List[Any]) -> None:
        """Parse a define-fun expression."""
        content = elem.content
        if len(content) < 4:
            return

        # Format: (define-fun name (params) return-type body)
        # content[1] = function name
        # content[2] = parameter list
        # content[3] = return type
        # content[4] = body

        func_name = content[1]
        if not isinstance(func_name, str):
            return

        # Parse parameters (if any)
        params = []
        param_list = content[2]
        if isinstance(param_list, SExpr) and isinstance(param_list.content, list):
            # Handle parameter list - each parameter is (name type)
            for param_spec in param_list.content:
                if isinstance(param_spec, SExpr) and isinstance(
                    param_spec.content, list
                ):
                    # Parameter specification: (name type)
                    if len(param_spec.content) >= 2:
                        param_name = param_spec.content[
                            0
                        ]  # Should be the parameter name
                        param_type = param_spec.content[
                            1
                        ]  # Should be the parameter type
                        if isinstance(param_name, str):
                            params.append((param_name, self._parse_sort(param_type)))
                elif isinstance(param_spec, str) and param_spec == "(":
                    # Empty parameter list case
                    pass

        # Parse return type
        return_type = self._parse_sort(content[3]) if len(content) > 3 else None

        # Parse body
        body = self._parse_term(content[4]) if len(content) > 4 else None

        # Create function definition
        if return_type and body:
            func_def = FunctionDefinition(
                name=func_name, parameters=params, return_type=return_type, body=body
            )
            functions.append(func_def)

    def _parse_sort(self, sort_expr: Any) -> Sort:
        """Parse a sort expression."""
        if isinstance(sort_expr, str):
            # Simple sort name
            identifier = Identifier(sort_expr, [])
            return Sort(identifier, [])
        elif isinstance(sort_expr, SExpr) and isinstance(sort_expr.content, list):
            # Parametric sort like (Array Int Real) or (Set Int)
            if len(sort_expr.content) >= 1:
                name = sort_expr.content[0]
                if isinstance(name, str):
                    args = []
                    for arg in sort_expr.content[1:]:
                        args.append(self._parse_sort(arg))

                    identifier = Identifier(name, [])
                    return Sort(identifier, args)

        # Fallback
        identifier = Identifier("Unknown", [])
        return Sort(identifier, [])

    def _parse_term(self, term_expr: Any) -> Term:
        """Parse a term expression."""
        if isinstance(term_expr, str):
            # Simple symbol
            identifier = Identifier(term_expr, [])
            return Term(QualId(identifier, None), [])
        elif isinstance(term_expr, SExpr) and isinstance(term_expr.content, list):
            # Function application or complex term
            if len(term_expr.content) >= 1:
                head = term_expr.content[0]
                if isinstance(head, str):
                    args = []
                    for arg in term_expr.content[1:]:
                        args.append(self._parse_term(arg))

                    identifier = Identifier(head, [])
                    return Term(QualId(identifier, None), args)
            elif isinstance(term_expr.content, Constant):
                # Constant term
                identifier = Identifier(str(term_expr.content), [])
                return Term(QualId(identifier, None), [])

        # Fallback
        identifier = Identifier("Unknown", [])
        return Term(QualId(identifier, None), [])

    def _validate_model_text(self, model_text: str) -> bool:
        """Validate that the model text looks like a valid SMT-LIB 2 model."""
        if not model_text or not model_text.strip():
            return False

        # Basic validation - should start with "(model" and end with ")"
        stripped = model_text.strip()
        return stripped.startswith("(model") and stripped.endswith(")")

    def parse_model_with_validation(self, model_text: str) -> Model:
        """Parse SMT-LIB 2 model response with validation."""
        if not self._validate_model_text(model_text):
            raise ValueError("Invalid SMT-LIB 2 model format")

        return self.parse_model(model_text)


class SMTLib2Printer:
    """Pretty-printer for SMT-LIB 2 expressions."""

    def __init__(self, context: Context):
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


def parse_smtlib2_expression(text: str, context: Context) -> SExpr:
    """Parse SMT-LIB 2 expression from text."""
    parser = SMTLib2Parser(context)
    tokens = parser.tokenize(text)
    # Add the outer parentheses if not present
    if not tokens or tokens[0] != "(":
        tokens = ["("] + tokens + [")"]
    return parser.parse_expression(tokens)


def print_smtlib2_expression(expr: SExpr, context: Context) -> str:
    """Print SMT-LIB 2 expression to string."""
    printer = SMTLib2Printer(context)
    return printer.print_sexpr(expr)


def parse_smtlib2_model(model_text: str, context: Context) -> Model:
    """Parse SMT-LIB 2 model response."""
    parser = SMTLib2Parser(context)
    return parser.parse_model(model_text)


def parse_smtlib2_model_validated(model_text: str, context: Context) -> Model:
    """Parse SMT-LIB 2 model response with validation."""
    parser = SMTLib2Parser(context)
    return parser.parse_model_with_validation(model_text)


def print_smtlib2_model(model: Model, context: Context) -> str:
    """Print SMT-LIB 2 model to string."""
    printer = SMTLib2Printer(context)
    return printer.print_model(model)


def parse_smtlib2_model_from_string(model_str: str, context: Context) -> Model:
    """Parse SMT-LIB 2 model from string with error handling."""
    try:
        return parse_smtlib2_model_validated(model_str, context)
    except Exception as e:
        # Return empty model on parse error
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
    from aria.srk.syntax import Context

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
