"""
NeuroSym SMT-LIB2 parser — no external SMT solver dependency.
Supports QF_LIA, QF_BV, QF_ABV (single-query track subset).
"""

import re
from typing import List, Dict, Optional, Tuple, Any

from .ns_ast import (
    Sort, BoolSort, IntSort, BVSort, ArraySort,
    Term, BoolLit, IntLit, BVLit, Var, App,
    NsFormula,
    BOOL, INT, TRUE, FALSE,
    mk_and, mk_or, mk_not, mk_eq,
)


# ── Tokeniser ──────────────────────────────────────────────────────────────────

def _tokenise(s: str) -> List[Any]:
    """
    Returns a flat list of tokens:
      '('  ')'  str  ('bvlit', int, int)  ('kw', str)  ('str', str)
    Numbers are returned as int or str depending on context (always str here;
    the parser handles conversion).
    """
    tokens: List[Any] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == ';':                          # line comment
            while i < n and s[i] != '\n':
                i += 1
        elif c in ' \t\n\r':
            i += 1
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == '"':                        # string literal
            j = i + 1
            while j < n:
                if s[j] == '"':
                    if j + 1 < n and s[j + 1] == '"':
                        j += 2
                    else:
                        break
                else:
                    j += 1
            tokens.append(('str', s[i + 1:j]))
            i = j + 1
        elif c == '|':                        # quoted symbol
            j = s.index('|', i + 1)
            tokens.append(s[i + 1:j])
            i = j + 1
        elif c == '#':                        # BV literal
            if i + 1 < n and s[i + 1] == 'x':
                j = i + 2
                while j < n and s[j] in '0123456789abcdefABCDEF':
                    j += 1
                h = s[i + 2:j]
                tokens.append(('bvlit', int(h, 16), len(h) * 4))
                i = j
            elif i + 1 < n and s[i + 1] == 'b':
                j = i + 2
                while j < n and s[j] in '01':
                    j += 1
                b = s[i + 2:j]
                tokens.append(('bvlit', int(b, 2), len(b)))
                i = j
            else:
                i += 1
        elif c == ':':                        # keyword attribute
            j = i + 1
            while j < n and s[j] not in ' \t\n\r();':
                j += 1
            tokens.append(('kw', s[i:j]))
            i = j
        else:                                 # symbol / numeral
            j = i
            while j < n and s[j] not in ' \t\n\r();"#|':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


# ── Token stream helper ────────────────────────────────────────────────────────

class _Stream:
    def __init__(self, tokens: list):
        self._t = tokens
        self._i = 0

    def peek(self) -> Any:
        if self._i < len(self._t):
            return self._t[self._i]
        return None

    def pop(self) -> Any:
        t = self._t[self._i]
        self._i += 1
        return t

    def expect(self, val):
        t = self.pop()
        if t != val:
            raise ParseError(f"expected {val!r}, got {t!r}")
        return t

    def at_end(self) -> bool:
        return self._i >= len(self._t)


class ParseError(Exception):
    pass


# ── Sort parser ────────────────────────────────────────────────────────────────

def _parse_sort(st: _Stream) -> Sort:
    t = st.peek()
    if t == '(':
        st.pop()  # '('
        head = st.pop()
        if head == '_':
            # (_ BitVec n)  or  (_ FloatingPoint e s) etc.
            kind = st.pop()
            if kind == 'BitVec':
                w = int(st.pop())
                st.expect(')')
                return BVSort(w)
            else:
                # skip unknown indexed sort
                depth = 1
                while depth:
                    x = st.pop()
                    if x == '(': depth += 1
                    elif x == ')': depth -= 1
                return BVSort(1)  # placeholder
        elif head == 'Array':
            idx = _parse_sort(st)
            elem = _parse_sort(st)
            st.expect(')')
            return ArraySort(idx, elem)
        else:
            # Unknown compound sort — skip
            depth = 1
            while depth:
                x = st.pop()
                if x == '(': depth += 1
                elif x == ')': depth -= 1
            return INT
    else:
        name = st.pop()
        if name == 'Int':   return INT
        if name == 'Bool':  return BOOL
        if name == 'Real':  return INT  # approximate
        return INT  # unknown sort


# ── Term parser ────────────────────────────────────────────────────────────────

# Map SMT-LIB2 operators → sort inference rules
# 'bool'  → result is Bool
# 'arg0'  → result sort equals first argument sort
# 'concat'→ special
_BOOL_OPS = {
    'and', 'or', 'not', 'xor', '=>', 'implies',
    '=', 'distinct',
    '<', '<=', '>', '>=',
    'bvult', 'bvule', 'bvugt', 'bvuge',
    'bvslt', 'bvsle', 'bvsgt', 'bvsge',
}

_ARG0_OPS = {
    'bvadd', 'bvsub', 'bvmul',
    'bvsdiv', 'bvudiv', 'bvsrem', 'bvurem', 'bvsmod',
    'bvand', 'bvor', 'bvxor', 'bvnand', 'bvnor', 'bvxnor',
    'bvshl', 'bvlshr', 'bvashr',
    'bvneg', 'bvnot',
    '+', '-', '*', 'div', 'mod', 'abs', 'rem',
    'select',  # array select
}


def _infer_sort(op: str, params: tuple, args: List[Term]) -> Sort:
    if op in _BOOL_OPS:
        return BOOL
    if op == 'bvcomp':
        return BVSort(1)
    if op == 'concat':
        wa = args[0].sort.width if isinstance(args[0].sort, BVSort) else 0
        wb = args[1].sort.width if isinstance(args[1].sort, BVSort) else 0
        return BVSort(wa + wb)
    if op == 'extract':
        hi, lo = params
        return BVSort(hi - lo + 1)
    if op in ('zero_extend', 'sign_extend', 'repeat'):
        n = params[0]
        if op == 'repeat':
            w = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
            return BVSort(w * n)
        w = args[0].sort.width if isinstance(args[0].sort, BVSort) else 0
        return BVSort(w + n)
    if op in ('rotate_left', 'rotate_right'):
        return args[0].sort if args else BVSort(1)
    if op == 'store':
        return args[0].sort if args else ArraySort(INT, INT)
    if op == 'ite':
        return args[1].sort if len(args) > 1 else BOOL
    if op in _ARG0_OPS and args:
        return args[0].sort
    if args:
        return args[0].sort
    return BOOL


def _parse_term(st: _Stream, env: Dict[str, Term]) -> Term:
    t = st.peek()

    # BV literal token
    if isinstance(t, tuple) and t[0] == 'bvlit':
        st.pop()
        return BVLit(t[1], t[2])

    # String / keyword — skip attribute values
    if isinstance(t, tuple) and t[0] in ('str', 'kw'):
        st.pop()
        return TRUE

    # Atom: true, false, numeral, symbol
    if t != '(':
        st.pop()
        if t == 'true':  return TRUE
        if t == 'false': return FALSE
        # Negative numeral presented as two tokens in some files; here it's
        # always a single string starting with digit or '-'
        if isinstance(t, str):
            # pure integer
            try:
                return IntLit(int(t))
            except ValueError:
                pass
            # variable reference or let-bound name
            if t in env:
                return env[t]
            # unknown symbol — return a Bool placeholder
            return Var(t, BOOL)
        return TRUE

    # Compound S-expression: ( head ... )
    st.pop()  # '('
    head = st.peek()

    # Empty list — treat as true
    if head == ')':
        st.pop()
        return TRUE

    # ── let binding ─────────────────────────────────────────────────────────
    if head == 'let':
        st.pop()  # 'let'
        st.expect('(')
        bindings = {}
        while st.peek() != ')':
            st.expect('(')
            name = st.pop()
            val  = _parse_term(st, env)
            st.expect(')')
            bindings[name] = val
        st.expect(')')
        new_env = {**env, **bindings}
        body = _parse_term(st, new_env)
        st.expect(')')
        return body

    # ── forall / exists — skip quantifiers (not QF) ─────────────────────────
    if head in ('forall', 'exists'):
        st.pop()
        # skip variable list
        st.expect('(')
        depth = 1
        while depth:
            x = st.pop()
            if x == '(': depth += 1
            elif x == ')': depth -= 1
        body = _parse_term(st, env)
        st.expect(')')
        return body

    # ── indexed operator: (_ op params...) ─────────────────────────────────
    if head == '(':
        # peek inside for '_'
        saved_i = st._i
        st.pop()  # inner '('
        inner = st.peek()
        if inner == '_':
            st.pop()  # '_'
            op     = st.pop()
            params = []
            while st.peek() != ')':
                params.append(int(st.pop()))
            st.expect(')')  # close inner '('
            params = tuple(params)
            # now parse arguments
            args = []
            while st.peek() != ')':
                args.append(_parse_term(st, env))
            st.expect(')')  # close outer '('
            sort = _infer_sort(op, params, args)
            return App(op, args, sort, params)
        else:
            # Not an indexed op — it's a nested expression used as function head
            # Restore and fall through to treat whole thing as expression
            st._i = saved_i

    # ── special forms ────────────────────────────────────────────────────────
    if head == '_':
        # (_ bvNNN w) — BV numeral with explicit width
        st.pop()  # '_'
        sym = st.pop()
        if isinstance(sym, str) and sym.startswith('bv'):
            val = int(sym[2:])
            w   = int(st.pop())
            st.expect(')')
            return BVLit(val, w)
        # other indexed constants (e.g., +oo, -oo for FP) — placeholder
        while st.peek() != ')':
            st.pop()
        st.expect(')')
        return TRUE

    if head == 'ite':
        st.pop()
        cond  = _parse_term(st, env)
        then_ = _parse_term(st, env)
        else_ = _parse_term(st, env)
        st.expect(')')
        sort = then_.sort
        return App('ite', [cond, then_, else_], sort)

    if head == 'as':
        # (as const (Array ...)) — array constant
        st.pop()
        inner = _parse_term(st, env)
        sort  = _parse_sort(st)
        st.expect(')')
        return App('as_const', [inner], sort)

    # ── general operator application ─────────────────────────────────────────
    op = st.pop()
    if not isinstance(op, str):
        # skip malformed
        while st.peek() != ')':
            st.pop()
        st.pop()
        return TRUE

    # Operator might itself be an indexed op: ((_ extract 7 0) x)
    if op == '(':
        inner_op = st.peek()
        if inner_op == '_':
            st.pop()  # '_'
            real_op = st.pop()
            params  = []
            while st.peek() != ')':
                params.append(int(st.pop()))
            st.expect(')')
            params = tuple(params)
            args   = []
            while st.peek() != ')':
                args.append(_parse_term(st, env))
            st.expect(')')
            sort = _infer_sort(real_op, params, args)
            return App(real_op, args, sort, params)
        else:
            # unusual; skip
            while st.peek() != ')':
                st.pop()
            st.pop()
            return TRUE

    args: List[Term] = []
    while st.peek() != ')':
        args.append(_parse_term(st, env))
    st.expect(')')

    sort = _infer_sort(op, (), args)
    return App(op, args, sort)


# ── Top-level command parser ───────────────────────────────────────────────────

def _collect_vars(term: Term, seen: Dict[str, Var]):
    if isinstance(term, Var):
        seen[term.name] = term
    elif isinstance(term, App):
        for a in term.args:
            _collect_vars(a, seen)


def parse_string(smtlib_str: str) -> NsFormula:
    tokens     = _tokenise(smtlib_str)
    st         = _Stream(tokens)
    logic      = 'UNKNOWN'
    decl_vars: Dict[str, Var] = {}
    assertions: List[Term]    = []
    defs:       Dict[str, Term] = {}

    while not st.at_end():
        if st.peek() != '(':
            st.pop()
            continue
        st.pop()  # '('
        cmd = st.pop()

        if cmd == 'set-logic':
            logic = st.pop()
            st.expect(')')

        elif cmd == 'declare-fun':
            name = st.pop()
            # parameter sorts (always empty for QF)
            st.expect('(')
            param_sorts = []
            while st.peek() != ')':
                param_sorts.append(_parse_sort(st))
            st.expect(')')
            ret_sort = _parse_sort(st)
            st.expect(')')
            if not param_sorts:
                v = Var(name, ret_sort)
                decl_vars[name] = v
            # function declarations with params: skip for now

        elif cmd == 'declare-const':
            name     = st.pop()
            ret_sort = _parse_sort(st)
            st.expect(')')
            v = Var(name, ret_sort)
            decl_vars[name] = v

        elif cmd == 'define-fun':
            name = st.pop()
            st.expect('(')
            params = []
            while st.peek() != ')':
                st.expect('(')
                pname = st.pop()
                psort = _parse_sort(st)
                st.expect(')')
                params.append((pname, psort))
            st.expect(')')
            _ret_sort = _parse_sort(st)
            body = _parse_term(st, {**decl_vars, **defs,
                                    **{p: Var(p, s) for p, s in params}})
            st.expect(')')
            if not params:
                defs[name] = body  # constant definition

        elif cmd == 'assert':
            env  = {**decl_vars, **defs}
            term = _parse_term(st, env)
            st.expect(')')
            assertions.append(term)

        elif cmd in ('check-sat', 'get-model', 'exit',
                     'get-value', 'get-unsat-core'):
            # skip to closing paren
            while st.peek() != ')':
                st.pop()
            st.pop()

        elif cmd in ('set-info', 'set-option', 'push', 'pop'):
            # skip
            depth = 1
            while depth:
                x = st.pop()
                if x == '(': depth += 1
                elif x == ')': depth -= 1

        else:
            # unknown command — skip to matching ')'
            depth = 1
            while depth:
                x = st.pop()
                if x == '(': depth += 1
                elif x == ')': depth -= 1

    # Collect any variables referenced in assertions but not declared
    for a in assertions:
        _collect_vars(a, decl_vars)

    var_names = sorted(decl_vars.keys())
    return NsFormula(
        logic      = str(logic),
        assertions = assertions,
        variables  = decl_vars,
        var_names  = var_names,
        source     = smtlib_str,
    )


def parse_file(path: str) -> NsFormula:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return parse_string(f.read())
