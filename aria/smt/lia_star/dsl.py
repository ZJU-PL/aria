from z3 import *

cards = set([])
setofs = set([])
empties = set([])
ms_unions = set([])
set_unions = set([])
ms_inters = set([])
set_subtracts = set([])
ms_subtracts = set([])
ms_subsets = set([])

def MS(A):
    return ArraySort(A, IntSort())

def is_ms_sort(s):
    return Z3_get_sort_kind(s.ctx.ref(), s.ast) == Z3_ARRAY_SORT and IntSort() == s.range()

def is_ms_card(t):
    global cards
    return is_app(t) and t.decl() in cards

def is_setof(t):
    global setofs
    return is_app(t) and t.decl() in setofs

def is_ms_empty(t):
    global empties
    return t in empties

def is_ms_union(t):
    global ms_unions
    return is_app(t) and t.decl() in ms_unions

def is_set_union(t):
    global set_unions
    return is_app(t) and t.decl() in set_unions

def is_ms_inter(t):
    global ms_inters
    return is_app(t) and t.decl() in ms_inters

def is_set_subtract(t):
    global set_subtracts
    return is_app(t) and t.decl() in set_subtracts

def is_ms_subtract(t):
    global ms_subtracts
    return is_app(t) and t.decl() in ms_subtracts

def is_ms_subset(t):
    global ms_subsets
    return is_app(t) and t.decl() in ms_subsets

def is_ms_var(v):
    return is_app(v) and v.num_args() == 0 and v.decl().kind() == Z3_OP_UNINTERPRETED and is_ms_sort(v.sort())

def card(ms):
    global cards
    c = Function('card', ms.sort(), IntSort())
    cards |= { c }
    return c(ms)

def setof(ms):
    global setofs
    assert (isinstance(ms.sort(), ArraySortRef))
    c = Function('setof', ms.sort(), ms.sort())
    setofs |= { c }
    return c(ms)

def empty(A):
    global empties
    e = K(A, IntVal(0))
    empties |= { e }
    return e

def U(S1, S2):
    global set_unions
    u = Function('Union', S1.sort(), S2.sort(), S1.sort())
    set_unions |= { u }
    return u(S1, S2)

def MU(S1, S2):
    global ms_unions
    u = Function('Union', S1.sort(), S2.sort(), S1.sort())
    ms_unions |= { u }
    return u(S1, S2)

def I(S1, S2):
    global ms_inters
    i = Function('Intersect', S1.sort(), S2.sort(), S1.sort())
    ms_inters |= { i }
    return i(S1, S2)

def SetSubtract(S1, S2):
    global set_subtracts
    s = Function('\\', S1.sort(), S2.sort(), S1.sort())
    set_subtracts |= { s }
    return s(S1, S2)

def MsSubtract(S1, S2):
    global ms_subtracts
    s = Function('\\\\', S1.sort(), S2.sort(), S1.sort())
    ms_subtracts |= { s }
    return s(S1, S2)

def MsSubset(S1, S2):
    global ms_subsets
    s = Function('MsSubset', S1.sort(), S2.sort(), BoolSort())
    ms_subsets |= { s }
    return s(S1, S2)


class LiaStar:
    def __init__(self):
        self.star_defs = []
        self.visited = {}
        self.vars = {}
        self.star_fmls = []

    def convert(self, fml):
        fml = self.visit(fml)
        # fml & (us in Sum_{ms_vars} ds)
        return fml, self.star_defs, self.star_fmls

    def fresh_var(self, name):
        return FreshConst(IntSort(), name)

    def add_star_def(self, d):
        u = self.fresh_var("u")
        # track that u is the output of d*
        self.star_defs += [(u, d)]
        return u

    def ms2var(self, t):
        if t in self.vars:
            return self.vars[t]
        v = self.fresh_var(t.decl().name())
        self.vars[t] = v
        self.star_fmls += [v >= 0]
        return v

    def visit(self, t):
        if t in self.visited:
            return self.visited[t]
        r = self.visit1(t)
        self.visited[t] = r;
        return r

    def visit1(self, t):
        chs = [self.visit(f) for f in t.children()]
        if is_and(t):
            return And(chs)
        if is_or(t):
            return Or(chs)
        if is_not(t):
            return Not(chs[0])
        if is_ms_card(t):
            return self.add_star_def(chs[0])
        if is_setof(t):
            return If(chs[0] > 0, 1, 0)
        if is_ms_empty(t):
            return 0
        if is_ms_union(t):
            return chs[0] + chs[1]
        if is_set_union(t):
            return If(Or(chs[0] > 0, chs[1] > 0), 1, 0)
        if is_ms_inter(t):
            t1 = chs[0]
            t2 = chs[1]
            return If(t1 >= t2, t2, t1)
        if is_ms_subtract(t):
            t1 = chs[0]
            t2 = chs[1]
            return If(t2 == 0, t1, 0)
        if is_set_subtract(t):
            t1 = chs[0]
            t2 = chs[1]
            return If(t1 <= t2, 0, t1 - t2)
        if is_ms_var(t):
            return self.ms2var(t)
        if is_ms_subset(t):
            u = self.add_star_def(If(chs[0] > chs[1], 1, 0))
            return u == 0
        if is_eq(t) and is_ms_sort(t.arg(0).sort()):
            u = self.add_star_def(If(chs[0] == chs[1], 0, 1))
            return u == 0
        if is_app(t):
            return t.decl()(chs)
        assert (False)
        return None

def to_lia_star(fml):
    ls = LiaStar()
    return ls.convert(fml)

mapa_flag = False

class Bapa2Ms:
    def __init__(self):
        self.visited = {}
        self.set2ms_vars = {}

    def fresh_var(self, s, name):
        return FreshConst(s, name)

    def convert(self, fmls):
        fmls = [self.visit(fml) for fml in fmls]
        return fmls

    def visit(self, t):
        if t in self.visited:
            return self.visited[t]
        r = self.visit1(t)
        self.visited[t] = r;
        return r

    def is_set_sort(self, s):
        return Z3_get_sort_kind(s.ctx.ref(), s.ast) == Z3_ARRAY_SORT and BoolSort() == s.range()

    def is_set_var(self, t):
        return is_app(t) and t.num_args() == 0 and t.decl().kind() == Z3_OP_UNINTERPRETED and self.is_set_sort(t.sort())

    def set2ms(self, t):
        if t in self.set2ms_vars:
            return self.set2ms_vars[t]
        A = t.sort().domain()
        v = self.fresh_var(MS(A), "%s" % t)
        self.set2ms_vars[t] = v
        return v

    def is_set_card(self, t):
        return is_app(t) and t.num_args() == 1 and t.decl().name() == 'card'

    def is_set_union(self, t):
        return is_app(t) and t.num_args() == 2 and t.decl().kind() == Z3_OP_SET_UNION

    def is_set_eq(self, t):
        return is_eq(t) and self.is_set_sort(t.arg(0).sort())

    def is_set_subtract(self, t):
        return is_app(t) and t.decl().kind() == Z3_OP_SET_DIFFERENCE

    def is_set_inter(self, t):
        return is_app(t) and t.decl().kind() == Z3_OP_SET_INTERSECT

    def is_set_empty(self, t):
        return is_app(t) and t.decl().kind() == Z3_OP_CONST_ARRAY and is_false(t.arg(0))

    def is_set_subset(self, t):
        return is_app(t) and t.decl().kind() == Z3_OP_SET_SUBSET

    # convert sets to multi-sets.
    def visit1(self, t):
        global mapa_flag
        chs = [self.visit(f) for f in t.children()]
        if self.is_set_var(t):
            return self.set2ms(t)
        if self.is_set_card(t):
            if mapa_flag:
                return card(chs[0])
            else:
                return card(setof(chs[0]))
        if self.is_set_union(t):
            assert len(chs) == 2
            return U(chs[0], chs[1])
        if self.is_set_inter(t):
            assert len(chs) == 2
            return I(chs[0], chs[1])
        if self.is_set_subtract(t):
            assert len(chs) == 2
            return SetSubtract(chs[0], chs[1])
        if self.is_set_empty(t):
            return empty(t.sort().domain(0))
        if self.is_set_subset(t):
            assert len(chs) == 2
            return MsSubset(chs[0], chs[1])
        if self.is_set_eq(t):
            return setof(chs[0]) == setof(chs[1])
        if is_and(t):
            return And(chs)
        if is_or(t):
            return Or(chs)
        if is_app(t):
            return t.decl()(chs)
        assert (False)
        return None

# Perform conversion on the given formulas
def bapa2ms(fmls):
    b2ms = Bapa2Ms()
    return b2ms.convert(fmls)

# Parse BAPA file, convert to multi-set formula
def parse_bapa(file, mapa):
    global mapa_flag

    # Read file into solver state
    s = Solver()
    s.from_file(file)

    # Set flag for parsing bapa examples as multiset formulas vs set formulas
    if mapa:
        mapa_flag = True

    # Conversion
    return bapa2ms(s.assertions())