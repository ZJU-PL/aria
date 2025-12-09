"""
Utilities for parsing and manipulating S-expressions.

S-expressions are symbolic expressions that represent nested lists of atoms,
commonly used in Lisp-like languages. This module provides tools to parse
string representations of S-expressions into Python data structures.

Example:
    "(+ 1 (* 2 3))" -> ['+', 1, ['*', 2, 3]]
"""

import json
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass

# Type definitions
Symbol = str
Number = Union[int, float]
Atom = Union[Symbol, Number]
SExpr = Union[Atom, List['SExpr']]


class SExprParser:
    """
    A class for parsing and manipulating S-expressions.
    """
    
    @dataclass
    class ParseError(Exception):
        """Exception raised for S-expression parsing errors."""
        message: str
        position: int
        expression: str

        def __str__(self) -> str:
            return f"{self.message} at position {self.position}: {self.expression}"

    @staticmethod
    def tokenize(expression: str) -> List[str]:
        """
        Convert a string into a list of S-expression tokens.

        Args:
            expression: String containing S-expression

        Returns:
            List of tokens (parentheses, atoms, etc.)

        Example:
            >>> SExprParser.tokenize("(+ 1 2)")
            ['(', '+', '1', '2', ')']
        """
        return (expression.replace('(', ' ( ')
                .replace(')', ' ) ')
                .replace('" "', 'space')
                .split())

    @classmethod
    def parse(cls, expression: str) -> Optional[SExpr]:
        """
        Parse a string into an S-expression.

        Args:
            expression: String containing S-expression

        Returns:
            Parsed S-expression as nested Python lists/atoms

        Raises:
            ParseError: If the expression is malformed

        Example:
            >>> SExprParser.parse("(+ 1 (* 2 3))")
            ['+', 1, ['*', 2, 3]]
        """
        tokens = cls.tokenize(expression)
        if not tokens:
            return None
        try:
            result, remaining = cls._parse_tokens(tokens, 0)
            if remaining:
                raise cls.ParseError("Unexpected trailing tokens", len(tokens) - len(remaining), expression)
            return result
        except (IndexError, ValueError) as e:
            raise cls.ParseError(str(e), len(tokens), expression)

    @classmethod
    def _parse_tokens(cls, tokens: List[str], depth: int) -> Tuple[SExpr, List[str]]:
        """
        Recursively parse a list of tokens into an S-expression.

        Args:
            tokens: List of remaining tokens to parse
            depth: Current nesting depth

        Returns:
            Tuple of (parsed expression, remaining tokens)
        """
        if not tokens:
            raise ValueError("Unexpected end of expression")

        token = tokens[0]
        remaining = tokens[1:]

        if token == '(':
            lst: List[SExpr] = []
            while remaining and remaining[0] != ')':
                expr, remaining = cls._parse_tokens(remaining, depth + 1)
                lst.append(expr)
            if not remaining:
                raise ValueError("Missing closing parenthesis")
            return lst, remaining[1:]  # Skip closing paren

        elif token == ')':
            raise ValueError("Unexpected closing parenthesis")

        else:
            return cls.parse_atom(token), remaining

    @staticmethod
    def parse_atom(token: str) -> Atom:
        """
        Convert a token string into an atomic value (number or symbol).

        Args:
            token: String token to parse

        Returns:
            Parsed atomic value
        """
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return Symbol(token)

    @classmethod
    def parse_sexpr_string(cls, program: str) -> Optional[SExpr]:
        """
        Read an S-expression from a string.
        This is an alias for parse().

        Args:
            program: String containing S-expression

        Returns:
            Parsed S-expression as nested Python lists/atoms
        """
        return cls.parse(program)

    @staticmethod
    def sexpr_to_string(expr: SExpr) -> str:
        """
        Convert an S-expression to a string representation.

        Args:
            expr: The S-expression to convert

        Returns:
            String representation of the S-expression
        """
        if isinstance(expr, list):
            return f"({' '.join(SExprParser.sexpr_to_string(e) for e in expr)})"
        elif isinstance(expr, (int, float)):
            return str(expr)
        else:
            return str(expr)

    @classmethod
    def sexpr_to_json(cls, expr: SExpr) -> str:
        """
        Convert an S-expression to a JSON string.
        """
        return json.dumps(expr)
    
    @classmethod
    def sexpr_from_json(cls, json_str: str) -> SExpr:
        """
        Convert a JSON string into an S-expression.
        
        Args:
            json_str: A JSON string representing an S-expression
        
        Returns:
            The parsed S-expression data structure
            
        Raises:
            ValueError: If the JSON string is invalid or cannot be parsed
            TypeError: If the parsed JSON doesn't match the SExpr type structure
            
        Example:
            >>> SExprParser.sexpr_from_json('["+", 1, ["*", 2, 3]]')
            ['+', 1, ['*', 2, 3]]
        """
        try:
            parsed = json.loads(json_str)
            # Validate that the structure conforms to SExpr type
            cls._validate_sexpr(parsed)
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    
    @classmethod
    def _validate_sexpr(cls, expr) -> None:
        """
        Validate that a parsed object conforms to the SExpr type structure.
        
        Args:
            expr: The object to validate
            
        Raises:
            TypeError: If the object doesn't match the SExpr type structure
        """
        if isinstance(expr, (int, float, str)):
            # Atoms are valid SExpr values
            return
        elif isinstance(expr, list):
            # Recursively validate each element in the list
            for item in expr:
                cls._validate_sexpr(item)
        else:
            raise TypeError(f"Invalid SExpr type: {type(expr)}. Expected int, float, str, or list.")

