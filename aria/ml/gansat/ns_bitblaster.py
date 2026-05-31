"""
NeuroSym bit-blaster — converts QF_BV / QF_ABV formulas to SAT CNF.

Each BV variable of width w becomes w Boolean variables (MSB first is
index 0; index w-1 is LSB).  The blaster builds a circuit of AND/OR/XOR
gates encoded as Tseitin clauses.

Public API:
  blast(formula: NsFormula) → (clauses, n_vars, var_map)
    clauses  : list of list[int]  (signed, 1-indexed)
    n_vars   : total Boolean vars
    var_map  : dict {bv_var_name: list[int]} BV vars → bit-variable IDs (MSB first)
               bit-variable ID = SAT variable number (1-indexed)

After DPLL returns an assignment, call reconstruct(sat_assign, var_map) to
get the BV assignment dict.
"""

from typing import List, Dict, Tuple, Optional
from .ns_ast import (
    Term, BoolLit, IntLit, BVLit, Var, App,
    NsFormula, BoolSort, IntSort, BVSort, ArraySort,
    BOOL, TRUE, FALSE,
)


# ── Variable allocator ─────────────────────────────────────────────────────────

class _Alloc:
    def __init__(self):
        self._count = 0

    def fresh(self) -> int:
        self._count += 1
        return self._count

    def fresh_bits(self, w: int) -> List[int]:
        return [self.fresh() for _ in range(w)]

    @property
    def count(self) -> int:
        return self._count


# ── Tseitin gate builders ──────────────────────────────────────────────────────
# All functions return a NEW variable whose value equals the gate output.
# Clauses are appended to `out`.

def _new_eq_lit(a: int, out: list, alloc: _Alloc) -> int:
    """y = a  (just return a directly — no gate needed)."""
    return a


def _gate_not(a: int, out: list, alloc: _Alloc) -> int:
    return -a   # negate literal; no extra clause needed


def _gate_and(a: int, b: int, out: list, alloc: _Alloc) -> int:
    y = alloc.fresh()
    # y → (a ∧ b)   and   (a ∧ b) → y
    out.append([-y,  a])
    out.append([-y,  b])
    out.append([ y, -a, -b])
    return y


def _gate_or(a: int, b: int, out: list, alloc: _Alloc) -> int:
    y = alloc.fresh()
    out.append([ y, -a])
    out.append([ y, -b])
    out.append([-y,  a,  b])
    return y


def _gate_xor(a: int, b: int, out: list, alloc: _Alloc) -> int:
    y = alloc.fresh()
    out.append([-y, -a, -b])
    out.append([-y,  a,  b])
    out.append([ y, -a,  b])
    out.append([ y,  a, -b])
    return y


def _gate_ite(c: int, t: int, e: int, out: list, alloc: _Alloc) -> int:
    """y = ite(c, t, e)"""
    y = alloc.fresh()
    # y ↔ (c → t) ∧ (¬c → e)
    out.append([-y,  -c,  t])
    out.append([-y,   c,  e])
    out.append([ y,  -c, -t])
    out.append([ y,   c, -e])
    return y


def _gate_and_n(bits: List[int], out: list, alloc: _Alloc) -> int:
    """y = AND of all bits in list."""
    if not bits: return _CONST_TRUE(out, alloc)
    r = bits[0]
    for b in bits[1:]:
        r = _gate_and(r, b, out, alloc)
    return r


def _gate_or_n(bits: List[int], out: list, alloc: _Alloc) -> int:
    if not bits: return _CONST_FALSE(out, alloc)
    r = bits[0]
    for b in bits[1:]:
        r = _gate_or(r, b, out, alloc)
    return r


def _CONST_TRUE(out: list, alloc: _Alloc) -> int:
    v = alloc.fresh()
    out.append([v])
    return v


def _CONST_FALSE(out: list, alloc: _Alloc) -> int:
    v = alloc.fresh()
    out.append([-v])
    return v


# ── Bit-vector integer → bit list ──────────────────────────────────────────────

def _int_to_bits(val: int, w: int, out: list, alloc: _Alloc) -> List[int]:
    """Return a list of w constant literals (MSB first)."""
    result = []
    for i in range(w - 1, -1, -1):
        bit = (val >> i) & 1
        v   = alloc.fresh()
        if bit:
            out.append([v])
        else:
            out.append([-v])
        result.append(v)
    return result


# ── Adder circuit ──────────────────────────────────────────────────────────────

def _full_adder(a: int, b: int, cin: int,
                out: list, alloc: _Alloc) -> Tuple[int, int]:
    """Returns (sum_bit, carry_out)."""
    # sum  = a XOR b XOR cin
    ab   = _gate_xor(a,  b,   out, alloc)
    s    = _gate_xor(ab, cin, out, alloc)
    # cout = (a AND b) OR (cin AND (a XOR b))
    c1   = _gate_and(a, b,   out, alloc)
    c2   = _gate_and(cin, ab, out, alloc)
    cout = _gate_or(c1, c2,  out, alloc)
    return s, cout


def _bv_add(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> List[int]:
    """Ripple-carry adder. a_bits and b_bits are MSB-first."""
    w     = len(a_bits)
    carry = _CONST_FALSE(out, alloc)
    sums  = [0] * w
    for i in range(w - 1, -1, -1):
        s, carry = _full_adder(a_bits[i], b_bits[i], carry, out, alloc)
        sums[i] = s
    return sums   # MSB first; overflow carry discarded


def _bv_neg(a_bits: List[int], out: list, alloc: _Alloc) -> List[int]:
    """Two's complement negation: ~a + 1."""
    not_a = [_gate_not(b, out, alloc) for b in a_bits]
    one   = _int_to_bits(1, len(a_bits), out, alloc)
    return _bv_add(not_a, one, out, alloc)


def _bv_sub(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> List[int]:
    """a - b = a + (-b)."""
    neg_b = _bv_neg(b_bits, out, alloc)
    return _bv_add(a_bits, neg_b, out, alloc)


def _bv_and(a_bits, b_bits, out, alloc):
    return [_gate_and(a, b, out, alloc) for a, b in zip(a_bits, b_bits)]

def _bv_or(a_bits, b_bits, out, alloc):
    return [_gate_or(a, b, out, alloc) for a, b in zip(a_bits, b_bits)]

def _bv_xor(a_bits, b_bits, out, alloc):
    return [_gate_xor(a, b, out, alloc) for a, b in zip(a_bits, b_bits)]

def _bv_not(a_bits, out, alloc):
    return [_gate_not(b, out, alloc) for b in a_bits]


def _bv_mul(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> List[int]:
    """Schoolbook multiplication (w² AND gates). MSB first."""
    w     = len(a_bits)
    # Partial products
    result = _int_to_bits(0, w, out, alloc)
    for i in range(w - 1, -1, -1):
        # Shift a_bits left by (w-1-i) positions (= multiply by 2^(w-1-i))
        shift = w - 1 - i
        shifted = ([_CONST_FALSE(out, alloc)] * shift
                   + a_bits[:w - shift])     # MSB first, shift left
        # If b[i] is 1, add shifted to result
        masked = [_gate_and(shifted[j], b_bits[i], out, alloc)
                  for j in range(w)]
        result = _bv_add(result, masked, out, alloc)
    return result


# ── Comparators ────────────────────────────────────────────────────────────────

def _bv_eq(a_bits: List[int], b_bits: List[int],
           out: list, alloc: _Alloc) -> int:
    """Return single Boolean variable: 1 iff a == b."""
    eq_bits = [_gate_not(_gate_xor(a, b, out, alloc), out, alloc)
               for a, b in zip(a_bits, b_bits)]
    return _gate_and_n(eq_bits, out, alloc)


def _bv_ult(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> int:
    """Unsigned a < b."""
    # Compute a - b; if borrow occurred, a < b
    # Equivalently: NOT (a >= b) = NOT (b <= a)
    # Use subtraction: borrow = MSB carry-out of (a - b) is 1 → a < b
    # More directly: compute a + ~b + 1; if carry-in to MSB+1 is 0 → a < b
    # Simplest: propagate borrow bit from MSB
    w = len(a_bits)
    # borrow chain: b[i] = (a_i < b_i) OR (a_i == b_i AND borrow)
    borrow = _CONST_FALSE(out, alloc)
    for i in range(w - 1, -1, -1):
        ai, bi = a_bits[i], b_bits[i]
        # new_borrow = (NOT ai AND bi) OR (NOT (ai XOR bi) AND borrow)
        not_ai    = _gate_not(ai, out, alloc)
        ai_lt_bi  = _gate_and(not_ai, bi, out, alloc)
        eq_i      = _gate_not(_gate_xor(ai, bi, out, alloc), out, alloc)
        prop      = _gate_and(eq_i, borrow, out, alloc)
        borrow    = _gate_or(ai_lt_bi, prop, out, alloc)
    return borrow


def _bv_ule(a_bits, b_bits, out, alloc) -> int:
    """a <= b  iff  NOT (b < a)."""
    return _gate_not(_bv_ult(b_bits, a_bits, out, alloc), out, alloc)


def _bv_slt(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> int:
    """Signed a < b."""
    # If signs differ: a < b iff a is negative (MSB=1)
    # If signs equal:  unsigned compare of remaining bits
    w    = len(a_bits)
    a_s  = a_bits[0]        # sign bit of a
    b_s  = b_bits[0]        # sign bit of b
    # diff_sign = a_s AND NOT b_s  (a neg, b pos → a < b)
    not_b_s   = _gate_not(b_s, out, alloc)
    diff_sign = _gate_and(a_s, not_b_s, out, alloc)
    # same_sign = NOT (a_s XOR b_s)
    same_sign = _gate_not(_gate_xor(a_s, b_s, out, alloc), out, alloc)
    # ult_rest = unsigned compare
    ult_rest  = _bv_ult(a_bits, b_bits, out, alloc)
    # slt = diff_sign OR (same_sign AND ult_rest)
    both = _gate_and(same_sign, ult_rest, out, alloc)
    return _gate_or(diff_sign, both, out, alloc)


def _bv_sle(a_bits, b_bits, out, alloc) -> int:
    return _gate_not(_bv_slt(b_bits, a_bits, out, alloc), out, alloc)


# ── Shift circuits ─────────────────────────────────────────────────────────────

def _bv_shl(a_bits: List[int], b_bits: List[int],
            out: list, alloc: _Alloc) -> List[int]:
    """Logical shift left a by b (variable shift)."""
    w = len(a_bits)
    result = list(a_bits)
    for stage, bit in enumerate(reversed(b_bits)):  # LSB first
        shift_amt = 1 << stage
        if shift_amt >= w:
            # If this bit is set, all result bits are 0
            zero_bits = [_CONST_FALSE(out, alloc) for _ in range(w)]
            result = [_gate_ite(bit, zero_bits[i], result[i], out, alloc)
                      for i in range(w)]
            break
        shifted = result[shift_amt:] + [_CONST_FALSE(out, alloc)] * shift_amt
        result  = [_gate_ite(bit, shifted[i], result[i], out, alloc)
                   for i in range(w)]
    return result


def _bv_lshr(a_bits: List[int], b_bits: List[int],
             out: list, alloc: _Alloc) -> List[int]:
    """Logical shift right."""
    w = len(a_bits)
    result = list(a_bits)
    for stage, bit in enumerate(reversed(b_bits)):
        shift_amt = 1 << stage
        if shift_amt >= w:
            zero_bits = [_CONST_FALSE(out, alloc) for _ in range(w)]
            result = [_gate_ite(bit, zero_bits[i], result[i], out, alloc)
                      for i in range(w)]
            break
        shifted = [_CONST_FALSE(out, alloc)] * shift_amt + result[:w - shift_amt]
        result  = [_gate_ite(bit, shifted[i], result[i], out, alloc)
                   for i in range(w)]
    return result


def _bv_ashr(a_bits: List[int], b_bits: List[int],
             out: list, alloc: _Alloc) -> List[int]:
    """Arithmetic shift right (fill with sign bit)."""
    w    = len(a_bits)
    sign = a_bits[0]
    result = list(a_bits)
    for stage, bit in enumerate(reversed(b_bits)):
        shift_amt = 1 << stage
        if shift_amt >= w:
            fill = [sign] * w
            result = [_gate_ite(bit, fill[i], result[i], out, alloc)
                      for i in range(w)]
            break
        fill    = [sign] * shift_amt + result[:w - shift_amt]
        result  = [_gate_ite(bit, fill[i], result[i], out, alloc)
                   for i in range(w)]
    return result


# ── Main blaster ───────────────────────────────────────────────────────────────

class _Blaster:
    def __init__(self):
        self.alloc   = _Alloc()
        self.clauses: List[List[int]] = []
        self.var_map: Dict[str, List[int]] = {}
        self._cache: Dict[int, object] = {}   # id(term) → blasted value

    def _bv_var(self, name: str, width: int) -> List[int]:
        if name not in self.var_map:
            self.var_map[name] = self.alloc.fresh_bits(width)
        return self.var_map[name]

    def blast_bv(self, term: Term) -> List[int]:
        """Return list of SAT literals (MSB first) representing BV term."""
        tid = id(term)
        if tid in self._cache:
            return self._cache[tid]

        result = self._blast_bv_inner(term)
        self._cache[tid] = result
        return result

    def _blast_bv_inner(self, term: Term) -> List[int]:
        out    = self.clauses
        alloc  = self.alloc

        if isinstance(term, BVLit):
            return _int_to_bits(term.value, term.width, out, alloc)

        if isinstance(term, Var) and isinstance(term.sort, BVSort):
            return self._bv_var(term.name, term.sort.width)

        if not isinstance(term, App):
            return [_CONST_FALSE(out, alloc)]

        op, args, p = term.op, term.args, term._params
        w = term.sort.width if isinstance(term.sort, BVSort) else 1

        if op == 'bvadd':
            return _bv_add(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvsub':
            return _bv_sub(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvmul':
            return _bv_mul(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvneg':
            return _bv_neg(self.blast_bv(args[0]), out, alloc)
        if op == 'bvnot':
            return _bv_not(self.blast_bv(args[0]), out, alloc)
        if op == 'bvand':
            return _bv_and(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvor':
            return _bv_or(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvxor':
            return _bv_xor(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvnand':
            return _bv_not(_bv_and(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc), out, alloc)
        if op == 'bvnor':
            return _bv_not(_bv_or(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc), out, alloc)
        if op == 'bvxnor':
            return _bv_not(_bv_xor(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc), out, alloc)
        if op == 'bvshl':
            return _bv_shl(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvlshr':
            return _bv_lshr(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvashr':
            return _bv_ashr(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)

        if op == 'concat':
            return self.blast_bv(args[0]) + self.blast_bv(args[1])

        if op == 'extract':
            hi, lo = p[0], p[1]
            a_bits = self.blast_bv(args[0])
            total  = len(a_bits)
            # MSB first: bit index i from MSB = bit position (total-1-i) from LSB
            lo_idx = total - 1 - hi
            hi_idx = total - 1 - lo
            return a_bits[lo_idx : hi_idx + 1]

        if op == 'zero_extend':
            n      = p[0]
            a_bits = self.blast_bv(args[0])
            return [_CONST_FALSE(out, alloc)] * n + a_bits

        if op == 'sign_extend':
            n      = p[0]
            a_bits = self.blast_bv(args[0])
            sign   = a_bits[0]
            return [sign] * n + a_bits

        if op == 'rotate_left':
            n      = p[0] % w if w else 0
            a_bits = self.blast_bv(args[0])
            return a_bits[n:] + a_bits[:n]

        if op == 'rotate_right':
            n      = p[0] % w if w else 0
            a_bits = self.blast_bv(args[0])
            return a_bits[w - n:] + a_bits[:w - n]

        if op == 'repeat':
            n      = p[0]
            a_bits = self.blast_bv(args[0])
            return a_bits * n

        if op == 'bvcomp':
            eq = _bv_eq(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
            return [eq]

        if op == 'ite':
            cond  = self.blast_bool(args[0])
            t_    = self.blast_bv(args[1])
            e_    = self.blast_bv(args[2])
            return [_gate_ite(cond, t_[i], e_[i], out, alloc) for i in range(len(t_))]

        # Fallback: fresh unconstrained bits
        return self.alloc.fresh_bits(w)

    def blast_bool(self, term: Term) -> int:
        """Return a SAT literal for a Bool-sorted term."""
        tid = id(term)
        if tid in self._cache:
            return self._cache[tid]
        result = self._blast_bool_inner(term)
        self._cache[tid] = result
        return result

    def _blast_bool_inner(self, term: Term) -> int:
        out   = self.clauses
        alloc = self.alloc

        if isinstance(term, BoolLit):
            if term.value:
                return _CONST_TRUE(out, alloc)
            else:
                return _CONST_FALSE(out, alloc)

        if isinstance(term, Var) and isinstance(term.sort, BoolSort):
            return self._bv_var(term.name, 1)[0]

        if not isinstance(term, App):
            return _CONST_TRUE(out, alloc)

        op, args = term.op, term.args

        if op == 'and':
            lits = [self.blast_bool(a) for a in args]
            return _gate_and_n(lits, out, alloc)
        if op == 'or':
            lits = [self.blast_bool(a) for a in args]
            return _gate_or_n(lits, out, alloc)
        if op == 'not':
            return _gate_not(self.blast_bool(args[0]), out, alloc)
        if op == 'xor':
            r = self.blast_bool(args[0])
            for a in args[1:]:
                r = _gate_xor(r, self.blast_bool(a), out, alloc)
            return r
        if op in ('=>', 'implies'):
            a_ = self.blast_bool(args[0])
            b_ = self.blast_bool(args[1])
            return _gate_or(_gate_not(a_, out, alloc), b_, out, alloc)
        if op == 'ite':
            c_ = self.blast_bool(args[0])
            t_ = self.blast_bool(args[1])
            e_ = self.blast_bool(args[2])
            return _gate_ite(c_, t_, e_, out, alloc)

        if op == '=':
            s0 = args[0].sort
            if isinstance(s0, BVSort):
                return _bv_eq(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
            elif isinstance(s0, BoolSort):
                a_ = self.blast_bool(args[0])
                b_ = self.blast_bool(args[1])
                return _gate_not(_gate_xor(a_, b_, out, alloc), out, alloc)
            else:
                return _CONST_TRUE(out, alloc)

        if op == 'distinct':
            s0 = args[0].sort
            if isinstance(s0, BVSort) and len(args) == 2:
                eq = _bv_eq(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
                return _gate_not(eq, out, alloc)
            return _CONST_TRUE(out, alloc)

        # BV comparisons
        if op == 'bvult':
            return _bv_ult(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvule':
            return _bv_ule(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvugt':
            return _bv_ult(self.blast_bv(args[1]), self.blast_bv(args[0]), out, alloc)
        if op == 'bvuge':
            return _bv_ule(self.blast_bv(args[1]), self.blast_bv(args[0]), out, alloc)
        if op == 'bvslt':
            return _bv_slt(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvsle':
            return _bv_sle(self.blast_bv(args[0]), self.blast_bv(args[1]), out, alloc)
        if op == 'bvsgt':
            return _bv_slt(self.blast_bv(args[1]), self.blast_bv(args[0]), out, alloc)
        if op == 'bvsge':
            return _bv_sle(self.blast_bv(args[1]), self.blast_bv(args[0]), out, alloc)

        if op == 'bvcomp':
            bits = self.blast_bv(term)
            return bits[0]

        return _CONST_TRUE(out, alloc)


# ── Public API ─────────────────────────────────────────────────────────────────

def blast(formula: NsFormula):
    """
    Bit-blast a QF_BV / QF_ABV formula.
    Returns (clauses, n_vars, var_map).
    """
    blaster = _Blaster()
    top_lits = []

    for assertion in formula.assertions:
        lit = blaster.blast_bool(assertion)
        top_lits.append(lit)

    # Assert all top-level literals to be True
    for lit in top_lits:
        blaster.clauses.append([lit])

    return blaster.clauses, blaster.alloc.count, blaster.var_map


def reconstruct(sat_assign: Dict[int, bool],
                var_map: Dict[str, List[int]]) -> Dict[str, int]:
    """
    Convert SAT assignment back to BV variable values.
    sat_assign: {sat_var (1-indexed): bool}
    var_map:    {bv_var_name: [sat_var, ...]}  (MSB first)
    """
    result = {}
    for name, bits in var_map.items():
        value = 0
        for bit_var in bits:
            bit_val = sat_assign.get(abs(bit_var), False)
            if bit_var < 0:
                bit_val = not bit_val
            value = (value << 1) | (1 if bit_val else 0)
        result[name] = value
    return result
