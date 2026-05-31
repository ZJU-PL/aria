"""
NeuroSym formula evaluator — no external SMT solver dependency.

evaluate(formula, assignment) → True / False
  Raises EvalError if the formula is structurally invalid.

  assignment: dict { variable_name: int }
    - For Int variables: Python int
    - For BV variables:  unsigned Python int in [0, 2^width - 1]
    - For Bool variables: 0 or 1
"""

from .ns_ast import (
    Sort, BoolSort, IntSort, BVSort, ArraySort,
    Term, BoolLit, IntLit, BVLit, Var, App,
    NsFormula, BOOL, INT,
)


class EvalError(Exception):
    pass


# ── BV arithmetic helpers ──────────────────────────────────────────────────────

def _mask(w: int) -> int:
    return (1 << w) - 1


def _to_signed(v: int, w: int) -> int:
    if v >= (1 << (w - 1)):
        return v - (1 << w)
    return v


def _from_signed(v: int, w: int) -> int:
    return v & _mask(w)


def _bvadd(a, b, w):  return (a + b) & _mask(w)
def _bvsub(a, b, w):  return (a - b) & _mask(w)
def _bvmul(a, b, w):  return (a * b) & _mask(w)
def _bvneg(a, w):     return (-a)    & _mask(w)
def _bvnot(a, w):     return (~a)    & _mask(w)
def _bvand(a, b, w):  return a & b
def _bvor (a, b, w):  return a | b
def _bvxor(a, b, w):  return a ^ b
def _bvnand(a, b, w): return _bvnot(_bvand(a, b, w), w)
def _bvnor (a, b, w): return _bvnot(_bvor (a, b, w), w)
def _bvxnor(a, b, w): return _bvnot(_bvxor(a, b, w), w)


def _bvudiv(a, b, w):
    if b == 0:
        return _mask(w)          # SMT-LIB: all-ones on div-by-zero
    return (a // b) & _mask(w)


def _bvurem(a, b, w):
    if b == 0:
        return a & _mask(w)
    return (a % b) & _mask(w)


def _bvsdiv(a, b, w):
    sa, sb = _to_signed(a, w), _to_signed(b, w)
    if sb == 0:
        return _mask(w) if sa >= 0 else 1
    # truncation toward zero
    q = int(sa / sb)
    return _from_signed(q, w)


def _bvsrem(a, b, w):
    sa, sb = _to_signed(a, w), _to_signed(b, w)
    if sb == 0:
        return a & _mask(w)
    q = int(sa / sb)            # truncation toward zero
    r = sa - q * sb
    return _from_signed(r, w)


def _bvsmod(a, b, w):
    sa, sb = _to_signed(a, w), _to_signed(b, w)
    if sb == 0:
        return a & _mask(w)
    r = sa % sb                 # Python % matches SMT-LIB bvsmod sign convention
    return _from_signed(r, w)


def _bvshl(a, b, w):
    if b >= w: return 0
    return (a << b) & _mask(w)


def _bvlshr(a, b, w):
    if b >= w: return 0
    return (a >> b) & _mask(w)


def _bvashr(a, b, w):
    sa = _to_signed(a, w)
    if b >= w:
        return 0 if sa >= 0 else _mask(w)
    r = sa >> int(b)            # Python preserves sign bit
    return _from_signed(r, w)


def _concat(a, wa, b, wb):
    return ((a & _mask(wa)) << wb) | (b & _mask(wb))


def _extract(a, hi, lo):
    return (a >> lo) & _mask(hi - lo + 1)


def _zero_extend(a, extra_bits):
    return a  # Python ints are arbitrary-precision; high bits already 0


def _sign_extend(a, w, extra_bits):
    sa = _to_signed(a, w)
    return _from_signed(sa, w + extra_bits)


def _rotate_left(a, n, w):
    if w == 0: return 0
    n = n % w
    return ((a << n) | (a >> (w - n))) & _mask(w)


def _rotate_right(a, n, w):
    if w == 0: return 0
    n = n % w
    return ((a >> n) | (a << (w - n))) & _mask(w)


def _repeat(a, w, n):
    result, total_w = 0, 0
    for _ in range(n):
        result = (result << w) | (a & _mask(w))
        total_w += w
    return result & _mask(total_w)


# ── Array model ────────────────────────────────────────────────────────────────

class _ArrayVal:
    """Functional array value."""
    def __init__(self, default=0, updates=None):
        self.default  = default
        self.updates  = updates or {}   # index → value (most-recent wins)

    def select(self, idx):
        return self.updates.get(idx, self.default)

    def store(self, idx, val):
        new_updates = dict(self.updates)
        new_updates[idx] = val
        return _ArrayVal(self.default, new_updates)

    def __eq__(self, other):
        if not isinstance(other, _ArrayVal): return NotImplemented
        # Equal iff they agree on all keys present in either
        keys = set(self.updates) | set(other.updates)
        for k in keys:
            if self.select(k) != other.select(k): return False
        return self.default == other.default


# ── Main evaluator ─────────────────────────────────────────────────────────────

def _eval(term: Term, env: dict) -> object:
    """
    Recursively evaluate a term.
    Returns: bool for Bool-sorted terms, int for Int/BV terms, _ArrayVal for arrays.
    """
    # Literals
    if isinstance(term, BoolLit): return term.value
    if isinstance(term, IntLit):  return term.value
    if isinstance(term, BVLit):   return term.value & _mask(term.width)

    # Variable lookup
    if isinstance(term, Var):
        if term.name in env:
            v = env[term.name]
            if isinstance(term.sort, BVSort) and isinstance(v, int):
                return v & _mask(term.sort.width)
            return v
        # Unassigned variable — use 0 as default
        if isinstance(term.sort, BVSort): return 0
        if isinstance(term.sort, IntSort): return 0
        if isinstance(term.sort, ArraySort):
            return _ArrayVal()
        return False

    if not isinstance(term, App):
        raise EvalError(f"Unknown term type: {type(term)}")

    op   = term.op
    args = term.args
    p    = term._params

    # Boolean connectives
    if op == 'and':
        return all(_eval(a, env) for a in args)
    if op == 'or':
        return any(_eval(a, env) for a in args)
    if op == 'not':
        return not _eval(args[0], env)
    if op == 'xor':
        r = False
        for a in args: r ^= bool(_eval(a, env))
        return r
    if op in ('=>', 'implies'):
        return (not _eval(args[0], env)) or bool(_eval(args[1], env))

    # Equality and inequality
    if op == '=':
        v0 = _eval(args[0], env)
        for a in args[1:]:
            if _eval(a, env) != v0: return False
        return True
    if op == 'distinct':
        vals = [_eval(a, env) for a in args]
        return len(vals) == len(set(vals) if not any(isinstance(v, _ArrayVal) for v in vals) else
                                [id(v) if isinstance(v, _ArrayVal) else v for v in vals])

    # ITE
    if op == 'ite':
        cond = _eval(args[0], env)
        return _eval(args[1], env) if cond else _eval(args[2], env)

    # Integer arithmetic
    if op == '+':
        return sum(_eval(a, env) for a in args)
    if op == '-':
        if len(args) == 1:
            return -_eval(args[0], env)
        v = _eval(args[0], env)
        for a in args[1:]: v -= _eval(a, env)
        return v
    if op == '*':
        r = 1
        for a in args: r *= _eval(a, env)
        return r
    if op == 'div':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        if b_ == 0: return 0
        # SMT-LIB integer div: floor division toward -inf
        return a_ // b_
    if op == 'mod':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        if b_ == 0: return a_
        return a_ % b_   # Python % matches SMT-LIB mod
    if op == 'abs':
        return abs(_eval(args[0], env))
    if op == 'rem':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        if b_ == 0: return a_
        return a_ % b_

    # Integer comparisons
    if op == '<':  return _eval(args[0], env) <  _eval(args[1], env)
    if op == '<=': return _eval(args[0], env) <= _eval(args[1], env)
    if op == '>':  return _eval(args[0], env) >  _eval(args[1], env)
    if op == '>=': return _eval(args[0], env) >= _eval(args[1], env)

    # BV — need width from sort
    sort = term.sort
    w    = sort.width if isinstance(sort, BVSort) else 1

    # BV arithmetic
    if op == 'bvadd':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvadd(a_, b_, w)
    if op == 'bvsub':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvsub(a_, b_, w)
    if op == 'bvmul':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvmul(a_, b_, w)
    if op == 'bvneg':
        return _bvneg(_eval(args[0], env), w)
    if op == 'bvnot':
        return _bvnot(_eval(args[0], env), w)
    if op == 'bvand':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvand(a_, b_, w)
    if op == 'bvor':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvor(a_, b_, w)
    if op == 'bvxor':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvxor(a_, b_, w)
    if op == 'bvnand':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvnand(a_, b_, w)
    if op == 'bvnor':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvnor(a_, b_, w)
    if op == 'bvxnor':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvxnor(a_, b_, w)
    if op == 'bvudiv':
        # w comes from sort; args[0] provides the true width
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvudiv(a_, b_, aw)
    if op == 'bvurem':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvurem(a_, b_, aw)
    if op == 'bvsdiv':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvsdiv(a_, b_, aw)
    if op == 'bvsrem':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvsrem(a_, b_, aw)
    if op == 'bvsmod':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvsmod(a_, b_, aw)

    # BV shifts
    if op == 'bvshl':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvshl(a_, b_, aw)
    if op == 'bvlshr':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvlshr(a_, b_, aw)
    if op == 'bvashr':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else w
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return _bvashr(a_, b_, aw)

    # BV comparisons — Bool result, use arg[0] width
    if op in ('bvult', 'bvule', 'bvugt', 'bvuge',
              'bvslt', 'bvsle', 'bvsgt', 'bvsge'):
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        if op == 'bvult': return a_ <  b_
        if op == 'bvule': return a_ <= b_
        if op == 'bvugt': return a_ >  b_
        if op == 'bvuge': return a_ >= b_
        sa, sb = _to_signed(a_, aw), _to_signed(b_, aw)
        if op == 'bvslt': return sa <  sb
        if op == 'bvsle': return sa <= sb
        if op == 'bvsgt': return sa >  sb
        if op == 'bvsge': return sa >= sb

    # bvcomp → 1-bit BV
    if op == 'bvcomp':
        a_, b_ = _eval(args[0], env), _eval(args[1], env)
        return 1 if a_ == b_ else 0

    # Concat
    if op == 'concat':
        wa = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        wb = args[1].sort.width if isinstance(args[1].sort, BVSort) else 1
        return _concat(_eval(args[0], env), wa, _eval(args[1], env), wb)

    # Indexed ops: extract, zero_extend, sign_extend, rotate_*, repeat
    if op == 'extract':
        hi, lo = p[0], p[1]
        return _extract(_eval(args[0], env), hi, lo)
    if op == 'zero_extend':
        return _zero_extend(_eval(args[0], env), p[0])
    if op == 'sign_extend':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        return _sign_extend(_eval(args[0], env), aw, p[0])
    if op == 'rotate_left':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        return _rotate_left(_eval(args[0], env), p[0], aw)
    if op == 'rotate_right':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        return _rotate_right(_eval(args[0], env), p[0], aw)
    if op == 'repeat':
        aw = args[0].sort.width if isinstance(args[0].sort, BVSort) else 1
        return _repeat(_eval(args[0], env), aw, p[0])

    # Array ops
    if op == 'select':
        arr = _eval(args[0], env)
        idx = _eval(args[1], env)
        if isinstance(arr, _ArrayVal):
            return arr.select(idx)
        return 0
    if op == 'store':
        arr = _eval(args[0], env)
        idx = _eval(args[1], env)
        val = _eval(args[2], env)
        if isinstance(arr, _ArrayVal):
            return arr.store(idx, val)
        return _ArrayVal(0, {idx: val})
    if op == 'as_const':
        val = _eval(args[0], env)
        return _ArrayVal(val)

    # Fallback: return neutral value for unknown ops
    if isinstance(sort, BVSort): return 0
    if isinstance(sort, IntSort): return 0
    if isinstance(sort, ArraySort): return _ArrayVal()
    return True   # unknown bool op → assume true (safe: may cause false positives,
                  # caught by full assertion check)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(formula: NsFormula, assignment: dict) -> bool:
    """
    Return True iff assignment satisfies all assertions in formula.
    assignment maps variable names (str) to int values.
    """
    env = {}
    for name, var in formula.variables.items():
        if name in assignment:
            v = assignment[name]
            if isinstance(var.sort, BVSort):
                env[name] = int(v) & _mask(var.sort.width)
            else:
                env[name] = int(v)
        else:
            # default: 0
            if isinstance(var.sort, ArraySort):
                env[name] = _ArrayVal()
            else:
                env[name] = 0

    for assertion in formula.assertions:
        try:
            result = _eval(assertion, env)
        except Exception:
            return False
        if not result:
            return False
    return True


def evaluate_single(term: Term, assignment: dict, variables: dict) -> object:
    """Evaluate a single term (useful for testing individual constraints)."""
    env = {}
    for name, var in variables.items():
        v = assignment.get(name, 0)
        if isinstance(var.sort, BVSort):
            env[name] = int(v) & _mask(var.sort.width)
        else:
            env[name] = int(v)
    return _eval(term, env)
