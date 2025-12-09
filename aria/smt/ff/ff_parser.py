#!/usr/bin/env python3
"""
ff_parser.py  –  Tiny SMT-LIB parser for the theory of Finite Fields
-------------------------------------------------------------------
Only the fragments that appear in QF_FF benchmarks are implemented.
If you meet a new construct the parser will raise  SyntaxError(...)
so you immediately see what to extend.
"""
from __future__ import annotations
import re, pathlib
from typing import List, Dict, Tuple, Any, Union
from .ff_ast import (
    FieldExpr, FieldAdd, FieldMul, FieldNeg, FieldEq, FieldVar, FieldConst,
    FieldSub, FieldPow, FieldDiv, BoolOr, BoolAnd, BoolNot, BoolImplies, BoolIte, BoolVar, ParsedFormula
)

Token = str
Sexp  = Union[Token, List['Sexp']]

token_re = re.compile(r"\(|\)|[^\s()]+")

CONST_HASH_RE = re.compile(r"#f(\d+)m(\d+)")
CONST_AS_RE   = re.compile(r"ff(\d+)")

class FFParserError(Exception):
    pass

def tokenize(txt:str)->List[Token]:
    # Remove comments (lines starting with ;)
    lines = txt.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove comment portion if present
        comment_pos = line.find(';')
        if comment_pos >= 0:
            line = line[:comment_pos]
        cleaned_lines.append(line)
    txt = '\n'.join(cleaned_lines)
    return token_re.findall(txt)

def parse_sexp(tokens:List[Token], idx:int=0)->Tuple[Sexp,int]:
    if idx >= len(tokens):
        raise FFParserError("unexpected EOF")
    tok = tokens[idx]
    if tok == '(':  # list
        lst: List[Sexp] = []
        idx += 1
        while idx < len(tokens) and tokens[idx] != ')':
            elem, idx = parse_sexp(tokens, idx)
            lst.append(elem)
        if idx >= len(tokens):
            raise FFParserError("unmatched '('")
        return lst, idx+1  # skip ')'
    elif tok == ')':
        raise FFParserError("unmatched ')'")
    else:
        return tok, idx+1

def parse_file(path:str)->List[Sexp]:
    txt = pathlib.Path(path).read_text()
    tokens = tokenize(txt)
    sexps: List[Sexp] = []
    idx = 0
    while idx < len(tokens):
        sx, idx = parse_sexp(tokens, idx)
        sexps.append(sx)
    return sexps

# ---------------- interpretation to AST -----------------------------

def build_formula(sexps:List[Sexp])->ParsedFormula:
    p: int | None = None
    sort_alias: Dict[str,int] = {}
    variables: Dict[str,str] = {}
    assertions: List[FieldExpr] = []
    expected_status: str | None = None

    def ensure_p(newp:int):
        nonlocal p
        if p is None:
            p = newp
        elif p != newp:
            raise FFParserError(f"mixed field sizes {p} vs {newp}")

    def parse_constant(tok:Token, sort_name:str|None=None)->FieldConst:
        # try #fXmP
        m = CONST_HASH_RE.fullmatch(tok)
        if m:
            val = int(m.group(1))
            mod = int(m.group(2))
            ensure_p(mod)
            return FieldConst(val % mod)
        # generic ffN constant (ff0, ff1, ff2, etc.)
        m2 = CONST_AS_RE.fullmatch(tok)
        if m2:
            val = int(m2.group(1))
            if sort_name is None:
                raise FFParserError("ffN constant without sort context")
            mod = sort_alias.get(sort_name)
            if mod is None:
                raise FFParserError(f"unknown sort {sort_name}")
            ensure_p(mod)
            return FieldConst(val % mod)
        raise FFParserError(f"unrecognized constant {tok}")

    def interp(sx:Sexp, env:Dict[str,FieldExpr]) -> FieldExpr:
        # env for let bindings
        if isinstance(sx, str):
            if sx in env:
                return env[sx]
            if sx in variables:
                sort_type = variables[sx]
                if sort_type == 'bool':
                    return BoolVar(sx)
                else:
                    return FieldVar(sx)
            # maybe constant symbolic token? parse const if constant
            m = CONST_HASH_RE.fullmatch(sx)
            if m:
                return parse_constant(sx)
            if sx.startswith('ff'):
                # will resolve once we know sort? unsupported here
                raise FFParserError(f"bare ff constant {sx} not allowed")
            raise FFParserError(f"unknown symbol {sx}")
        # list
        if not sx:
            raise FFParserError("empty list")
        head = sx[0]
        if head == 'ff.add':
            return FieldAdd(*[interp(a, env) for a in sx[1:]])
        if head == 'ff.mul':
            return FieldMul(*[interp(a, env) for a in sx[1:]])
        if head == 'ff.neg':
            if len(sx)!=2:
                raise FFParserError("ff.neg takes 1 arg")
            return FieldNeg(interp(sx[1], env))
        if head == '=':
            if len(sx)!=3:
                raise FFParserError("= takes 2 args")
            return FieldEq(interp(sx[1], env), interp(sx[2], env))
        if head == 'or':
            return BoolOr(*[interp(a, env) for a in sx[1:]])
        if head == 'and':
            return BoolAnd(*[interp(a, env) for a in sx[1:]])
        if head == 'not':
            if len(sx)!=2:
                raise FFParserError("not takes 1 arg")
            return BoolNot(interp(sx[1], env))
        if head == '=>':
            if len(sx)!=3:
                raise FFParserError("=> takes 2 args")
            return BoolImplies(interp(sx[1], env), interp(sx[2], env))
        if head == 'ite':
            if len(sx)!=4:
                raise FFParserError("ite takes 3 args")
            return BoolIte(interp(sx[1], env), interp(sx[2], env), interp(sx[3], env))
        if head == 'let':
            # pattern: (let ((name val) ...) body)
            bindings = sx[1]
            body = sx[2]
            new_env = env.copy()
            for pair in bindings:
                if not (isinstance(pair, list) and len(pair)==2 and isinstance(pair[0], str)):
                    raise FFParserError("malformed let binding")
                new_env[pair[0]] = interp(pair[1], env)
            return interp(body, new_env)
        if head == 'as':
            # constant ascription: (as ff2 F)
            if len(sx)!=3:
                raise FFParserError("as form length")
            const_tok = sx[1]
            sort_tok  = sx[2]
            if isinstance(const_tok, str) and isinstance(sort_tok, str):
                return parse_constant(const_tok, sort_tok)
            raise FFParserError("unexpected as args")
        raise FFParserError(f"unsupported head {head}")

    for top in sexps:
        if not isinstance(top, list):
            continue
        if not top:
            continue
        tag = top[0]
        if tag == 'set-logic' or tag == 'set-option':
            continue
        if tag == 'set-info':
            # Extract status from (set-info :status 'unsat) or (set-info :status 'sat)
            if len(top) >= 3 and top[1] == ':status':
                status_val = top[2]
                # Handle quoted symbols - tokenizer produces "'unsat'" as a single token
                if isinstance(status_val, str):
                    # Remove surrounding quotes if present (both single and double quotes)
                    status_val = status_val.strip("'\"").strip()
                    if status_val in ('sat', 'unsat'):
                        expected_status = status_val
            continue
        if tag == 'define-sort':
            # (define-sort F () (_ FiniteField 5))
            name = top[1]
            sort_body = top[3]
            if isinstance(sort_body, list) and sort_body[0]=='_' and sort_body[1]=='FiniteField':
                mod = int(sort_body[2])
                sort_alias[name] = mod
                ensure_p(mod)
            continue
        if tag == 'declare-fun':
            # (declare-fun x () (_ FiniteField 5)) or (declare-fun a () Bool)
            vname = top[1]
            sort_body = top[3]
            if isinstance(sort_body, list) and sort_body[0]=='_' and sort_body[1]=='FiniteField':
                mod = int(sort_body[2])
                ensure_p(mod)
                variables[vname] = 'ff'
            elif isinstance(sort_body, str) and sort_body in sort_alias:
                ensure_p(sort_alias[sort_body])
                variables[vname]='ff'
            elif isinstance(sort_body, str) and sort_body == 'Bool':
                variables[vname] = 'bool'
            else:
                raise FFParserError("unsupported sort in declare-fun")
            continue
        if tag == 'assert':
            expr = top[1]
            assertions.append( interp(expr, {}) )
            continue
        if tag == 'check-sat':
            continue
        # ignore others
    if p is None:
        raise FFParserError("no finite field found")
    return ParsedFormula(p, variables, assertions, expected_status)

# convenience wrapper

def parse_ff_file(path:str)->ParsedFormula:
    sexps = parse_file(path)
    return build_formula(sexps)
