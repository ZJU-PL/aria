"""Parser for lightweight three-valued propositional formulas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .propositional import Formula


_ATOM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_TOKENS = (
    "<->",
    "<=>",
    "->",
    "⊼",
    "⊽",
    "⊕",
    "(",
    ")",
    "!",
    "&",
    "|",
    "^",
    "?",
    "¬",
    "∧",
    "∨",
    "→",
    "↔",
)


class ThreeValSyntaxError(ValueError):
    """Raised when a three-valued formula string is malformed."""


@dataclass(frozen=True)
class _Token:
    value: str
    position: int


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse(self) -> "Formula":
        formula = self._parse_biconditional()
        if self._peek() is not None:
            token = self._peek()
            assert token is not None
            raise ThreeValSyntaxError(
                f"Unexpected token {token.value!r} at position {token.position}"
            )
        return formula

    def _parse_biconditional(self) -> "Formula":
        from .propositional import Iff

        formula = self._parse_implication()
        while True:
            token = self._peek()
            if token is None or token.value not in {"<->", "<=>", "↔"}:
                return formula
            self._advance()
            formula = Iff(formula, self._parse_implication())

    def _parse_implication(self) -> "Formula":
        from .propositional import Implies

        left = self._parse_xor()
        token = self._peek()
        if token is not None and token.value in {"->", "→"}:
            self._advance()
            return Implies(left, self._parse_implication())
        return left

    def _parse_xor(self) -> "Formula":
        from .propositional import Xor

        formula = self._parse_disjunction()
        while True:
            token = self._peek()
            if token is None or token.value not in {"^", "xor", "⊕"}:
                return formula
            self._advance()
            formula = Xor(formula, self._parse_disjunction())

    def _parse_disjunction(self) -> "Formula":
        from .propositional import Nor, Or

        formula = self._parse_conjunction()
        while True:
            token = self._peek()
            if token is None:
                return formula
            if token.value in {"|", "∨", "or"}:
                self._advance()
                formula = Or(formula, self._parse_conjunction())
                continue
            if token.value in {"nor", "⊽"}:
                self._advance()
                formula = Nor(formula, self._parse_conjunction())
                continue
            return formula

    def _parse_conjunction(self) -> "Formula":
        from .propositional import And, Nand

        formula = self._parse_unary()
        while True:
            token = self._peek()
            if token is None:
                return formula
            if token.value in {"&", "∧", "and"}:
                self._advance()
                formula = And(formula, self._parse_unary())
                continue
            if token.value in {"nand", "⊼"}:
                self._advance()
                formula = Nand(formula, self._parse_unary())
                continue
            return formula

    def _parse_unary(self) -> "Formula":
        token = self._peek()
        if token is None:
            raise ThreeValSyntaxError("Unexpected end of input")

        unary_cls = _unary_tokens().get(token.value)
        if unary_cls is not None:
            self._advance()
            return unary_cls(self._parse_unary())

        if token.value == "(":
            self._advance()
            formula = self._parse_biconditional()
            closing = self._peek()
            if closing is None or closing.value != ")":
                raise ThreeValSyntaxError(
                    f"Expected ')' to close '(' at position {token.position}"
                )
            self._advance()
            return formula

        return self._parse_atom_or_constant()

    def _parse_atom_or_constant(self) -> "Formula":
        from .propositional import Variable

        constant_tokens = _constant_tokens()
        token = self._peek()
        if token is None:
            raise ThreeValSyntaxError("Unexpected end of input")

        self._advance()
        if token.value in constant_tokens:
            return constant_tokens[token.value]
        if _ATOM_PATTERN.fullmatch(token.value) is not None:
            return Variable(token.value)
        raise ThreeValSyntaxError(
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

        raise ThreeValSyntaxError(
            f"Unexpected character {char!r} at position {index}"
        )

    return tokens


def _constant_tokens():
    from .propositional import Constant, TruthValue

    return {
        "true": Constant(TruthValue.TRUE),
        "false": Constant(TruthValue.FALSE),
        "unknown": Constant(TruthValue.UNKNOWN),
        "⊤": Constant(TruthValue.TRUE),
        "⊥": Constant(TruthValue.FALSE),
        "?": Constant(TruthValue.UNKNOWN),
    }


def _unary_tokens():
    from .propositional import Not

    return {
        "!": Not,
        "not": Not,
        "¬": Not,
    }


def parse_formula(text: str) -> "Formula":
    """Parse a formula string into the three-valued propositional AST."""

    if not isinstance(text, str):
        raise TypeError(f"Three-valued formulas must be strings, got {text!r}")
    return _Parser(_tokenize(text)).parse()


__all__ = ["ThreeValSyntaxError", "parse_formula"]
