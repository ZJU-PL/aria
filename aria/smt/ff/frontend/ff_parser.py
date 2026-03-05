#!/usr/bin/env python3
"""
ff_parser.py  –  SMT-LIB parser for the theory of finite fields.

The parser intentionally focuses on the quantifier-free fragment used by the
local regression suite. It supports multiple finite-field sorts in the same
formula, expands non-recursive `define-fun` macros, and lowers a few theory
operators such as `ff.bitsum` into the core AST.
"""
from __future__ import annotations

import pathlib
import re
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

from ..core.ff_ast import (
    BOOL_SORT,
    BoolAnd,
    BoolConst,
    BoolIte,
    BoolImplies,
    BoolNot,
    BoolOr,
    BoolVar,
    BoolXor,
    FieldAdd,
    FieldConst,
    FieldDiv,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldSub,
    FieldVar,
    ParsedFormula,
    ff_sort_id,
    field_modulus_from_sort,
    infer_field_modulus,
    is_bool_sort,
)

Token = str
Sexp = Union[Token, List["Sexp"]]

token_re = re.compile(r"\(|\)|[^\s()]+")
CONST_HASH_RE = re.compile(r"#f(-?\d+)m(\d+)")
CONST_AS_RE = re.compile(r"ff(-?\d+)")


class FunctionDef(NamedTuple):
    """A lightweight non-recursive SMT-LIB function definition."""

    params: List[str]
    body: Sexp


class FFParserError(Exception):
    """Exception raised for parser errors."""


def tokenize(txt: str) -> List[Token]:
    """Tokenize SMT-LIB input text."""
    lines = txt.split("\n")
    cleaned_lines = []
    for line in lines:
        comment_pos = line.find(";")
        if comment_pos >= 0:
            line = line[:comment_pos]
        cleaned_lines.append(line)
    return token_re.findall("\n".join(cleaned_lines))


def parse_sexp(tokens: List[Token], idx: int = 0) -> Tuple[Sexp, int]:
    """Parse an s-expression from tokens."""
    if idx >= len(tokens):
        raise FFParserError("unexpected EOF")
    tok = tokens[idx]
    if tok == "(":
        lst = []
        idx += 1
        while idx < len(tokens) and tokens[idx] != ")":
            elem, idx = parse_sexp(tokens, idx)
            lst.append(elem)
        if idx >= len(tokens):
            raise FFParserError("unmatched '('")
        return lst, idx + 1
    if tok == ")":
        raise FFParserError("unmatched ')'")
    return tok, idx + 1


def parse_file(path: str) -> List[Sexp]:
    """Parse a file into a list of s-expressions."""
    txt = pathlib.Path(path).read_text(encoding="utf-8")
    tokens = tokenize(txt)
    sexps = []
    idx = 0
    while idx < len(tokens):
        sx, idx = parse_sexp(tokens, idx)
        sexps.append(sx)
    return sexps


def build_formula(
    sexps: List[Sexp],
) -> ParsedFormula:  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    """Build a ParsedFormula from a list of s-expressions."""
    sort_alias: Dict[str, str] = {}
    variables: Dict[str, str] = {}
    assertions: List[FieldExpr] = []
    expected_status: Optional[str] = None
    function_defs: Dict[str, FunctionDef] = {}
    field_sizes: Set[int] = set()

    def register_sort(sort_id: str) -> None:
        modulus = field_modulus_from_sort(sort_id)
        if modulus is not None:
            field_sizes.add(modulus)

    def parse_sort(sort_sx: Sexp) -> str:
        if isinstance(sort_sx, str):
            if sort_sx == "Bool":
                return BOOL_SORT
            if sort_sx in sort_alias:
                return sort_alias[sort_sx]
            raise FFParserError("unknown sort %s" % sort_sx)
        if (
            isinstance(sort_sx, list)
            and len(sort_sx) == 3
            and sort_sx[0] == "_"
            and sort_sx[1] == "FiniteField"
        ):
            return ff_sort_id(int(sort_sx[2]))
        raise FFParserError("unsupported sort %s" % sort_sx)

    def parse_constant(tok: Token, sort_sx: Optional[Sexp] = None) -> FieldConst:
        m = CONST_HASH_RE.fullmatch(tok)
        if m:
            modulus = int(m.group(2))
            field_sizes.add(modulus)
            return FieldConst(int(m.group(1)) % modulus, modulus)

        m = CONST_AS_RE.fullmatch(tok)
        if m:
            if sort_sx is None:
                raise FFParserError("ffN constant without sort context")
            sort_id = parse_sort(sort_sx)
            modulus = field_modulus_from_sort(sort_id)
            if modulus is None:
                raise FFParserError("ff constants require a finite-field sort")
            field_sizes.add(modulus)
            return FieldConst(int(m.group(1)) % modulus, modulus)

        raise FFParserError("unrecognized constant %s" % tok)

    def expand_macro(
        name: str, args: List[Sexp], env: Dict[str, FieldExpr]
    ) -> FieldExpr:
        if name not in function_defs:
            raise FFParserError("unknown function %s" % name)
        definition = function_defs[name]
        if len(args) != len(definition.params):
            raise FFParserError(
                "function %s expects %d arguments, got %d"
                % (name, len(definition.params), len(args))
            )
        call_env = dict(env)
        for param, arg in zip(definition.params, args):
            call_env[param] = interp(arg, env)
        return interp(definition.body, call_env)

    def lower_bitsum(args: List[FieldExpr]) -> FieldExpr:
        if not args:
            raise FFParserError("ff.bitsum requires at least one operand")
        modulus = infer_field_modulus(args[0], variables)
        if modulus is None:
            raise FFParserError("ff.bitsum requires finite-field arguments")

        terms = []
        coeff = 1
        for arg in args:
            arg_modulus = infer_field_modulus(arg, variables)
            if arg_modulus != modulus:
                raise FFParserError("ff.bitsum arguments must share a field sort")
            if coeff % modulus == 1:
                terms.append(arg)
            else:
                terms.append(FieldMul(FieldConst(coeff % modulus, modulus), arg))
            coeff = (coeff * 2) % modulus
        return FieldAdd(*terms)

    def interp(
        sx: Sexp, env: Dict[str, FieldExpr]
    ) -> FieldExpr:  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(sx, str):
            if sx in env:
                return env[sx]
            if sx in variables:
                if is_bool_sort(variables[sx]):
                    return BoolVar(sx)
                return FieldVar(sx)
            if sx in function_defs and not function_defs[sx].params:
                return expand_macro(sx, [], env)
            if sx == "true":
                return BoolConst(True)
            if sx == "false":
                return BoolConst(False)
            if CONST_HASH_RE.fullmatch(sx):
                return parse_constant(sx)
            if sx.startswith("ff"):
                raise FFParserError("bare ff constant %s not allowed" % sx)
            raise FFParserError("unknown symbol %s" % sx)

        if not sx:
            raise FFParserError("empty list")

        head = sx[0]
        if not isinstance(head, str):
            raise FFParserError("unsupported list head %s" % head)

        if head == "ff.add":
            return FieldAdd(*[interp(arg, env) for arg in sx[1:]])
        if head == "ff.mul":
            return FieldMul(*[interp(arg, env) for arg in sx[1:]])
        if head == "ff.neg":
            if len(sx) != 2:
                raise FFParserError("ff.neg takes 1 arg")
            return FieldNeg(interp(sx[1], env))
        if head == "ff.sub":
            if len(sx) < 3:
                raise FFParserError("ff.sub takes at least 2 args")
            return FieldSub(*[interp(arg, env) for arg in sx[1:]])
        if head == "ff.div":
            if len(sx) != 3:
                raise FFParserError("ff.div takes 2 args")
            return FieldDiv(interp(sx[1], env), interp(sx[2], env))
        if head == "ff.bitsum":
            return lower_bitsum([interp(arg, env) for arg in sx[1:]])
        if head == "=":
            if len(sx) != 3:
                raise FFParserError("= takes 2 args")
            return FieldEq(interp(sx[1], env), interp(sx[2], env))
        if head == "or":
            return BoolOr(*[interp(arg, env) for arg in sx[1:]])
        if head == "and":
            return BoolAnd(*[interp(arg, env) for arg in sx[1:]])
        if head == "xor":
            return BoolXor(*[interp(arg, env) for arg in sx[1:]])
        if head == "not":
            if len(sx) != 2:
                raise FFParserError("not takes 1 arg")
            return BoolNot(interp(sx[1], env))
        if head == "=>":
            if len(sx) != 3:
                raise FFParserError("=> takes 2 args")
            return BoolImplies(interp(sx[1], env), interp(sx[2], env))
        if head == "ite":
            if len(sx) != 4:
                raise FFParserError("ite takes 3 args")
            return BoolIte(interp(sx[1], env), interp(sx[2], env), interp(sx[3], env))
        if head == "let":
            if len(sx) != 3 or not isinstance(sx[1], list):
                raise FFParserError("malformed let expression")
            new_env = dict(env)
            for pair in sx[1]:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                ):
                    raise FFParserError("malformed let binding")
                new_env[pair[0]] = interp(pair[1], env)
            return interp(sx[2], new_env)
        if head == "as":
            if len(sx) != 3 or not isinstance(sx[1], str):
                raise FFParserError("malformed ascription")
            return parse_constant(sx[1], sx[2])
        if head in function_defs:
            return expand_macro(head, sx[1:], env)

        raise FFParserError("unsupported head %s" % head)

    for top in sexps:
        if not isinstance(top, list) or not top:
            continue
        tag = top[0]
        if tag in ("set-logic", "set-option", "check-sat", "get-model", "get-value"):
            continue
        if tag == "set-info":
            if len(top) >= 3 and top[1] == ":status" and isinstance(top[2], str):
                status_val = top[2].strip("'\"").strip()
                if status_val in ("sat", "unsat"):
                    expected_status = status_val
            continue
        if tag == "define-sort":
            if len(top) != 4 or not isinstance(top[1], str):
                raise FFParserError("malformed define-sort")
            if top[2] != []:
                raise FFParserError("parametric finite-field aliases are unsupported")
            sort_id = parse_sort(top[3])
            sort_alias[top[1]] = sort_id
            register_sort(sort_id)
            continue
        if tag in ("declare-fun", "declare-const"):
            if tag == "declare-fun":
                if len(top) != 4 or not isinstance(top[1], str):
                    raise FFParserError("malformed declare-fun")
                if top[2] != []:
                    raise FFParserError("only nullary declare-fun is supported")
                name = top[1]
                sort_sx = top[3]
            else:
                if len(top) != 3 or not isinstance(top[1], str):
                    raise FFParserError("malformed declare-const")
                name = top[1]
                sort_sx = top[2]
            sort_id = parse_sort(sort_sx)
            register_sort(sort_id)
            variables[name] = sort_id
            continue
        if tag == "define-fun":
            if len(top) != 5 or not isinstance(top[1], str) or not isinstance(top[2], list):
                raise FFParserError("malformed define-fun")
            params = []
            for param in top[2]:
                if (
                    not isinstance(param, list)
                    or len(param) != 2
                    or not isinstance(param[0], str)
                ):
                    raise FFParserError("malformed define-fun parameter")
                params.append(param[0])
                register_sort(parse_sort(param[1]))
            register_sort(parse_sort(top[3]))
            function_defs[top[1]] = FunctionDef(params=params, body=top[4])
            continue
        if tag == "assert":
            if len(top) != 2:
                raise FFParserError("malformed assert")
            assertions.append(interp(top[1], {}))
            continue

    if not field_sizes:
        raise FFParserError("no finite field found")

    only_field = sorted(field_sizes)[0] if len(field_sizes) == 1 else None
    return ParsedFormula(
        only_field,
        variables,
        assertions,
        expected_status=expected_status,
        field_sizes=sorted(field_sizes),
    )


def parse_ff_file(path: str) -> ParsedFormula:
    """Parse a finite-field SMT-LIB file into a ParsedFormula."""
    return build_formula(parse_file(path))


def parse_ff_file_strict(path: str) -> ParsedFormula:
    """Parse a file and require that it uses exactly one finite-field sort."""
    formula = parse_ff_file(path)
    formula.require_single_field()
    return formula
