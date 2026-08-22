"""Reusable, guarded symbolic models for common Python library operations."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import z3

from .model_registry import DEFAULT_MODEL_REGISTRY
from .models import StructuredValue, SymbolicTerm
from .symbolic import merge_guards, truthy


def _arguments(args: Sequence[SymbolicTerm]) -> list[SymbolicTerm]:
    if len(args) == 1 and isinstance(args[0].expression, StructuredValue):
        structured = args[0].expression
        if structured.kind in {"list", "tuple"}:
            return [structured.children[index] for index in sorted(structured.children)]
    return list(args)


def _sum(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    terms = _arguments(args)
    if any(
        term.expression is None or not z3.is_arith(term.expression) for term in terms
    ):
        return None
    expression = z3.IntVal(0)
    for term in terms:
        expression = expression + term.expression
    return SymbolicTerm(expression, merge_guards(*terms))


def _minimum(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    return _extreme(args, minimum=True)


def _maximum(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    return _extreme(args, minimum=False)


def _extreme(args: Sequence[SymbolicTerm], *, minimum: bool) -> Optional[SymbolicTerm]:
    terms = _arguments(args)
    if not terms or any(
        term.expression is None or not z3.is_arith(term.expression) for term in terms
    ):
        return None
    expression = terms[0].expression
    for term in terms[1:]:
        expression = z3.If(
            expression <= term.expression if minimum else expression >= term.expression,
            expression,
            term.expression,
        )
    return SymbolicTerm(expression, merge_guards(*terms))


def _all(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    terms = [truthy(term) for term in _arguments(args)]
    if any(term.expression is None for term in terms):
        return None
    return SymbolicTerm(
        z3.And(*[term.expression for term in terms]), merge_guards(*terms)
    )


def _any(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    terms = [truthy(term) for term in _arguments(args)]
    if any(term.expression is None for term in terms):
        return None
    return SymbolicTerm(
        z3.Or(*[term.expression for term in terms]), merge_guards(*terms)
    )


def _parse_int(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if (
        len(args) != 1
        or args[0].expression is None
        or not z3.is_string(args[0].expression)
    ):
        return None
    text = args[0].expression
    decimal = z3.InRe(text, z3.Plus(z3.Range("0", "9")))
    return SymbolicTerm(z3.StrToInt(text), (*args[0].guards, decimal))


def _format_int(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if (
        len(args) != 1
        or args[0].expression is None
        or not z3.is_int(args[0].expression)
    ):
        return None
    value = args[0].expression
    return SymbolicTerm(z3.IntToStr(value), (*args[0].guards, value >= 0))


def _ord(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if (
        len(args) != 1
        or args[0].expression is None
        or not z3.is_string(args[0].expression)
    ):
        return None
    text = args[0].expression
    return SymbolicTerm(z3.StrToCode(text), (*args[0].guards, z3.Length(text) == 1))


def _chr(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if (
        len(args) != 1
        or args[0].expression is None
        or not z3.is_int(args[0].expression)
    ):
        return None
    code = args[0].expression
    return SymbolicTerm(
        z3.StrFromCode(code),
        (*args[0].guards, code >= 0, code <= 0x10FFFF),
    )


def _pow(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if len(args) != 2 or any(term.expression is None for term in args):
        return None
    base, exponent = args[0].expression, args[1].expression
    if not z3.is_arith(base) or not z3.is_int_value(exponent):
        return None
    power = exponent.as_long()
    if power < 0:
        return None
    return SymbolicTerm(base**power, merge_guards(*args))


def _path_join(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if len(args) != 2 or any(
        term.expression is None or not z3.is_string(term.expression) for term in args
    ):
        return None
    left, right = args[0].expression, args[1].expression
    guards = (
        *merge_guards(*args),
        z3.Not(z3.SuffixOf(z3.StringVal("/"), left)),
        z3.Not(z3.PrefixOf(z3.StringVal("/"), right)),
    )
    return SymbolicTerm(z3.Concat(left, z3.StringVal("/"), right), guards)


def _path_basename(args: Sequence[SymbolicTerm]) -> Optional[SymbolicTerm]:
    if (
        len(args) != 1
        or args[0].expression is None
        or not z3.is_string(args[0].expression)
    ):
        return None
    path = args[0].expression
    return SymbolicTerm(
        path,
        (*args[0].guards, z3.Not(z3.Contains(path, z3.StringVal("/")))),
    )


PACKS: Dict[str, Dict[str, Callable[..., Optional[SymbolicTerm]]]] = {
    "collections": {
        "sum": _sum,
        "min": _minimum,
        "max": _maximum,
        "all": _all,
        "any": _any,
    },
    "parsing": {"int": _parse_int, "str": _format_int},
    "text": {"ord": _ord, "chr": _chr},
    "numeric": {"pow": _pow},
    "paths": {
        "join": _path_join,
        "posixpath.join": _path_join,
        "ntpath.join": _path_join,
        "basename": _path_basename,
        "posixpath.basename": _path_basename,
        "ntpath.basename": _path_basename,
    },
}


def available_model_packs() -> tuple[str, ...]:
    return tuple(sorted(PACKS))


def install_model_pack(name: str) -> None:
    try:
        functions = PACKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown symbolic model pack {name!r}") from exc
    for function_name, model in functions.items():
        DEFAULT_MODEL_REGISTRY.register_function(
            function_name,
            model,
            replace=True,
        )


def install_default_model_packs() -> None:
    for name in available_model_packs():
        install_model_pack(name)


install_default_model_packs()
