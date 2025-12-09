"""
SMT-LIB 2 definitions and types.

This module defines the AST types for SMT-LIB 2 responses,
particularly for the get-model command as specified in SMT-LIB 2.6.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from fractions import Fraction

from . import zZ
from . import qQ


@dataclass(frozen=True)
class Constant:
    """SMT-LIB 2 constant values."""
    value: Union[int, Fraction, str]

    @staticmethod
    def Int(zz: int) -> Constant:
        return Constant(zz)

    @staticmethod
    def Real(qq: Fraction) -> Constant:
        return Constant(qq)

    @staticmethod
    def String(s: str) -> Constant:
        return Constant(s)

    def __str__(self) -> str:
        if isinstance(self.value, int):
            return zZ.show(self.value)
        elif isinstance(self.value, Fraction):
            return qQ.show(self.value)
        else:
            return str(self.value)


Symbol = str


@dataclass(frozen=True)
class Index:
    """SMT-LIB 2 index - either numeric or symbolic."""
    value: Union[int, Symbol]

    @staticmethod
    def Num(zz: int) -> Index:
        return Index(zz)

    @staticmethod
    def Sym(sym: Symbol) -> Index:
        return Index(sym)

    def __str__(self) -> str:
        if isinstance(self.value, int):
            return zZ.show(self.value)
        else:
            return self.value


@dataclass(frozen=True)
class Identifier:
    """SMT-LIB 2 identifier - symbol with optional indices."""
    symbol: Symbol
    indices: Tuple[Index, ...]

    def __init__(self, symbol: Symbol, indices: List[Index]):
        object.__setattr__(self, 'symbol', symbol)
        object.__setattr__(self, 'indices', tuple(indices))

    def __str__(self) -> str:
        if not self.indices:
            return self.symbol
        else:
            indices_str = " ".join(str(idx) for idx in self.indices)
            return f"(_ {self.symbol} {indices_str})"


@dataclass(frozen=True)
class Sort:
    """SMT-LIB 2 sort - possibly parametric type."""
    identifier: Identifier
    arguments: Tuple[Sort, ...]

    def __init__(self, identifier: Identifier, arguments: List[Sort]):
        object.__setattr__(self, 'identifier', identifier)
        object.__setattr__(self, 'arguments', tuple(arguments))

    def __str__(self) -> str:
        if not self.arguments:
            return str(self.identifier)
        else:
            args_str = " ".join(str(arg) for arg in self.arguments)
            return f"({self.identifier} {args_str})"


@dataclass(frozen=True)
class QualId:
    """Qualified identifier with optional sort."""
    identifier: Identifier
    sort: Optional[Sort]

    def __str__(self) -> str:
        if self.sort is None:
            return str(self.identifier)
        else:
            return f"(as {self.identifier} {self.sort})"


@dataclass(frozen=True)
class Pattern:
    """Match pattern for quantifier matching."""
    constructor: Symbol
    arguments: Tuple[Symbol, ...]

    def __init__(self, constructor: Symbol, arguments: List[Symbol]):
        object.__setattr__(self, 'constructor', constructor)
        object.__setattr__(self, 'arguments', tuple(arguments))

    def __str__(self) -> str:
        if not self.arguments:
            return self.constructor
        else:
            args_str = " ".join(self.arguments)
            return f"({self.constructor} {args_str})"


@dataclass(frozen=True)
class SExpr:
    """S-expression in SMT-LIB 2."""
    content: Union[Constant, Symbol, str, List[SExpr]]

    @staticmethod
    def SConst(c: Constant) -> SExpr:
        return SExpr(c)

    @staticmethod
    def SSym(sym: Symbol) -> SExpr:
        return SExpr(sym)

    @staticmethod
    def SKey(kw: Symbol) -> SExpr:
        return SExpr(kw)

    @staticmethod
    def SSexpr(sexprs: List[SExpr]) -> SExpr:
        return SExpr(sexprs)

    def __str__(self) -> str:
        if isinstance(self.content, Constant):
            return str(self.content)
        elif isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            content_str = " ".join(str(item) for item in self.content)
            return f"({content_str})"
        else:
            return str(self.content)


@dataclass(frozen=True)
class AttributeValue:
    """Value of an attribute in SMT-LIB 2."""
    content: Union[Constant, Symbol, SExpr]

    @staticmethod
    def VConst(c: Constant) -> AttributeValue:
        return AttributeValue(c)

    @staticmethod
    def VSym(sym: Symbol) -> AttributeValue:
        return AttributeValue(sym)

    @staticmethod
    def VSexpr(sexpr: SExpr) -> AttributeValue:
        return AttributeValue(sexpr)

    def __str__(self) -> str:
        return str(self.content)


@dataclass(frozen=True)
class Attribute:
    """Attribute in SMT-LIB 2."""
    keyword: Symbol
    value: Optional[AttributeValue]

    def __str__(self) -> str:
        if self.value is None:
            return f":{self.keyword}"
        else:
            return f":{self.keyword} {self.value}"


@dataclass(frozen=True)
class Term:
    """SMT-LIB 2 term."""
    qual_id: QualId
    arguments: Tuple[Term, ...]

    def __init__(self, qual_id: QualId, arguments: List[Term]):
        object.__setattr__(self, 'qual_id', qual_id)
        object.__setattr__(self, 'arguments', tuple(arguments))

    def __str__(self) -> str:
        if not self.arguments:
            return str(self.qual_id)
        else:
            args_str = " ".join(str(arg) for arg in self.arguments)
            return f"({self.qual_id} {args_str})"


@dataclass(frozen=True)
class QuantifiedTerm:
    """Quantified SMT-LIB 2 term."""
    quantifier: str  # "forall" or "exists"
    variables: List[Tuple[Symbol, Sort]]
    body: Term

    def __str__(self) -> str:
        vars_str = " ".join(f"({var} {sort})" for var, sort in self.variables)
        return f"({self.quantifier} ({vars_str}) {self.body})"


@dataclass(frozen=True)
class LetTerm:
    """Let-binding term."""
    bindings: List[Tuple[Symbol, Term]]
    body: Term

    def __str__(self) -> str:
        bindings_str = " ".join(f"({var} {term})" for var, term in self.bindings)
        return f"(let ({bindings_str}) {self.body})"


@dataclass(frozen=True)
class LambdaTerm:
    """Lambda abstraction term."""
    variables: List[Tuple[Symbol, Sort]]
    body: Term

    def __str__(self) -> str:
        vars_str = " ".join(f"({var} {sort})" for var, sort in self.variables)
        return f"(lambda ({vars_str}) {self.body})"


@dataclass(frozen=True)
class FunctionDefinition:
    """Function definition in SMT-LIB 2 model."""
    name: str
    parameters: List[Tuple[str, Sort]]
    return_type: Sort
    body: Term

    def __str__(self) -> str:
        params_str = " ".join(f"({name} {sort})" for name, sort in self.parameters)
        return f"(define-fun {self.name} ({params_str}) {self.return_type} {self.body})"


@dataclass(frozen=True)
class SortDefinition:
    """Sort definition in SMT-LIB 2 model."""
    name: str
    arity: int

    def __str__(self) -> str:
        if self.arity == 0:
            return f"(declare-sort {self.name} 0)"
        else:
            return f"(declare-sort {self.name} {self.arity})"


@dataclass(frozen=True)
class Model:
    """SMT-LIB 2 model response."""
    functions: List[FunctionDefinition]
    sorts: List[SortDefinition]

    def __str__(self) -> str:
        return f"Model({len(self.functions)} functions, {len(self.sorts)} sorts)"

    def to_smtlib2_string(self) -> str:
        """Convert model to SMT-LIB 2 format string."""
        lines = ["(model"]
        for sort in self.sorts:
            lines.append(f"  {sort}")
        for func in self.functions:
            lines.append(f"  {func}")
        lines.append(")")
        return "\n".join(lines)


# Union type for all possible SMT-LIB 2 expressions
SMTLib2Expr = Union[Constant, Symbol, SExpr, Term, QuantifiedTerm, LetTerm, LambdaTerm, QualId, Sort, Pattern, Attribute, AttributeValue]
