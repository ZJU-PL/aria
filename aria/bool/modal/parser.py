"""Parser for lightweight modal formulas."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import List, Optional

# Lazy import to avoid cycle
def _Formula():
    from . import Formula
    return Formula


_ATOM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_TOKENS = (
    "[]",
    "<>",
    "->",
    "(",
    ")",
    "!",
    "&",
    "|",
    "¬",
    "∧",
    "∨",
    "→",
    "□",
    "◇",
    "⊤",
    "⊥",
)


class ModalSyntaxError(ValueError):
    """Raised when a modal formula string is malformed."""


@dataclass(frozen=True)
class _Token:
    value: str
    position: int


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse(self) -> "Formula":
        formula = self._parse_implication()
        if self._peek() is not None:
            token = self._peek()
            assert token is not None
            raise ModalSyntaxError(
                f"Unexpected token {token.value!r} at position {token.position}"
            )
        return formula

    def _parse_implication(self) -> "Formula":
        from . import Implies

        left = self._parse_disjunction()
        token = self._peek()
        if token is not None and token.value in {"->", "→"}:
            self._advance()
            return Implies(left, self._parse_implication())
        return left

    def _parse_disjunction(self) -> "Formula":
        from . import Or

        formula = self._parse_conjunction()
        while True:
            token = self._peek()
            if token is None or token.value not in {"|", "∨"}:
                return formula
            self._advance()
            formula = Or(formula, self._parse_conjunction())

    def _parse_conjunction(self) -> "Formula":
        from . import And

        formula = self._parse_unary()
        while True:
            token = self._peek()
            if token is None or token.value not in {"&", "∧"}:
                return formula
            self._advance()
            formula = And(formula, self._parse_unary())

    def _parse_unary(self) -> "Formula":
        unary_tokens = _unary_tokens()

        token = self._peek()
        if token is None:
            raise ModalSyntaxError("Unexpected end of input")

        unary_cls = unary_tokens.get(token.value)
        if unary_cls is not None:
            self._advance()
            return unary_cls(self._parse_unary())

        if token.value == "(":
            self._advance()
            formula = self._parse_implication()
            closing = self._peek()
            if closing is None or closing.value != ")":
                raise ModalSyntaxError(
                    f"Expected ')' to close '(' at position {token.position}"
                )
            self._advance()
            return formula

        return self._parse_atom_or_constant()

    def _parse_atom_or_constant(self) -> "Formula":
        from . import Atom

        constant_tokens = _constant_tokens()
        token = self._peek()
        if token is None:
            raise ModalSyntaxError("Unexpected end of input")

        self._advance()
        if token.value in constant_tokens:
            return constant_tokens[token.value]
        if _ATOM_PATTERN.fullmatch(token.value) is not None:
            return Atom(token.value)
        raise ModalSyntaxError(
            f"Unexpected token {token.value!r} at position {token.position}"
        )

    def _peek(self) -> Optional[_Token]:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        self._index += 1
        return token


def _tokenize(text: str) -> List[_Token]:
    tokens = []
    index = 0

    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue

        matched_symbol = next(
            (symbol for symbol in _SYMBOL_TOKENS if text.startswith(symbol, index)),
            None,
        )
        if matched_symbol is not None:
            tokens.append(_Token(matched_symbol, index))
            index += len(matched_symbol)
            continue

        atom_match = _ATOM_PATTERN.match(text, index)
        if atom_match is not None:
            token = atom_match.group(0)
            tokens.append(_Token(token, index))
            index = atom_match.end()
            continue

        raise ModalSyntaxError(
            f"Unexpected character {char!r} at position {index}"
        )

    return tokens


def _constant_tokens():
    from . import Constant

    return {
        "true": Constant(True),
        "false": Constant(False),
        "⊤": Constant(True),
        "⊥": Constant(False),
    }


def _unary_tokens():
    from . import Box, Diamond, Not

    return {
        "!": Not,
        "¬": Not,
        "[]": Box,
        "□": Box,
        "<>": Diamond,
        "◇": Diamond,
    }


def parse_formula(text: str) -> "Formula":
    """Parse a modal formula string into the existing modal AST."""

    if not isinstance(text, str):
        raise TypeError(f"Modal formulas must be strings, got {text!r}")

    return _Parser(_tokenize(text)).parse()
