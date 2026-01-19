"""
Domain Specific Language (DSL) module for BAPA/MAPA to LIA* translation.

This module provides tools to parse BAPA (Boolean Algebra with Presburger Arithmetic)
and MAPA (Multiset Algebra with Presburger Arithmetic) formulas and translate them
into a format suitable for the LIA* solver.

It handles:
1.  Identifying multiset and set operations in Z3 expressions.
2.  Translating Set operations to Multiset operations (via `Bapa2Ms`).
3.  Translating Multiset operations to LIA* "star definitions" (via `LiaStar`).

The LIA* translation separates the formula into:
-   A boolean skeleton with integer variables.
-   "Star definitions" that define these integer variables as summations over the domain elements.
"""

from z3 import *

# Global sets to track registered functions
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
    """Create a Multiset sort over domain A (Array A -> Int)."""
    return ArraySort(A, IntSort())


def is_ms_sort(s):
    """Check if a sort is a Multiset sort."""
    return (
        Z3_get_sort_kind(s.ctx.ref(), s.ast) == Z3_ARRAY_SORT and IntSort() == s.range()
    )


def is_ms_card(t):
    """Check if term is a multiset cardinality application."""
    global cards
    return is_app(t) and t.decl() in cards


def is_setof(t):
    """Check if term is a setof application."""
    global setofs
    return is_app(t) and t.decl() in setofs


def is_ms_empty(t):
    """Check if term is an empty multiset constant."""
    global empties
    return t in empties


def is_ms_union(t):
    """Check if term is a multiset union application."""
    global ms_unions
    return is_app(t) and t.decl() in ms_unions


def is_set_union(t):
    """Check if term is a set union application."""
    global set_unions
    return is_app(t) and t.decl() in set_unions


def is_ms_inter(t):
    """Check if term is a multiset intersection application."""
    global ms_inters
    return is_app(t) and t.decl() in ms_inters


def is_set_subtract(t):
    """Check if term is a set subtraction application."""
    global set_subtracts
    return is_app(t) and t.decl() in set_subtracts


def is_ms_subtract(t):
    """Check if term is a multiset subtraction application."""
    global ms_subtracts
    return is_app(t) and t.decl() in ms_subtracts


def is_ms_subset(t):
    """Check if term is a multiset subset application."""
    global ms_subsets
    return is_app(t) and t.decl() in ms_subsets


def is_ms_var(v):
    """Check if a variable is of Multiset sort."""
    return (
        is_app(v)
        and v.num_args() == 0
        and v.decl().kind() == Z3_OP_UNINTERPRETED
        and is_ms_sort(v.sort())
    )


def card(ms):
    """Create a cardinality term for a multiset."""
    global cards
    c = Function("card", ms.sort(), IntSort())
    cards |= {c}
    return c(ms)


def setof(ms):
    """Create a 'setof' term (characteristic function) for a multiset."""
    global setofs
    assert isinstance(ms.sort(), ArraySortRef)
    c = Function("setof", ms.sort(), ms.sort())
    setofs |= {c}
    return c(ms)


def empty(A):
    """Create an empty multiset (all counts 0) over domain A."""
    global empties
    e = K(A, IntVal(0))
    empties |= {e}
    return e


def U(S1, S2):
    """Create a set union term."""
    global set_unions
    u = Function("Union", S1.sort(), S2.sort(), S1.sort())
    set_unions |= {u}
    return u(S1, S2)


def MU(S1, S2):
    """Create a multiset union term."""
    global ms_unions
    u = Function("Union", S1.sort(), S2.sort(), S1.sort())
    ms_unions |= {u}
    return u(S1, S2)


def I(S1, S2):
    """Create a multiset intersection term."""
    global ms_inters
    i = Function("Intersect", S1.sort(), S2.sort(), S1.sort())
    ms_inters |= {i}
    return i(S1, S2)


def SetSubtract(S1, S2):
    """Create a set subtraction term."""
    global set_subtracts
    s = Function("\\", S1.sort(), S2.sort(), S1.sort())
    set_subtracts |= {s}
    return s(S1, S2)


def MsSubtract(S1, S2):
    """Create a multiset subtraction term."""
    global ms_subtracts
    s = Function("\\\\", S1.sort(), S2.sort(), S1.sort())
    ms_subtracts |= {s}
    return s(S1, S2)


def MsSubset(S1, S2):
    """Create a multiset subset term."""
    global ms_subsets
    s = Function("MsSubset", S1.sort(), S2.sort(), BoolSort())
    ms_subsets |= {s}
    return s(S1, S2)


class LiaStar:
    """
    Translates Multiset formulas to LIA* formulas.

    This class walks the AST of a formula and replaces multiset operations
    with corresponding integer arithmetic operations and "star definitions".
    Star definitions represent summations over the domain of the multiset.
    """

    def __init__(self):
        """
        Initializes a new LiaStar translator.
        """
        self.star_defs = []  # List of (int_var, definition_term)
        self.visited = {}    # Cache for visited nodes
        self.vars = {}       # Map from multiset vars to integer vars
        self.star_fmls = []  # Additional formulas (e.g., non-negativity)

    def convert(self, fml):
        """
        Convert a formula to LIA*.

        Args:
            fml: The Z3 formula to convert.

        Returns:
            Tuple (converted_fml, star_defs, star_fmls)
        """
        fml = self.visit(fml)
        # fml & (us in Sum_{ms_vars} ds)
        return fml, self.star_defs, self.star_fmls

    def fresh_var(self, name):
        """Create a fresh integer variable."""
        return FreshConst(IntSort(), name)

    def add_star_def(self, d):
        """
        Add a new star definition.

        Args:
            d: The definition term (representing the operation on a single element).

        Returns:
            The fresh integer variable 'u' representing sum(d(x) for x in Domain).
        """
        u = self.fresh_var("u")
        # track that u is the output of d*
        self.star_defs += [(u, d)]
        return u

    def ms2var(self, t: ExprRef) -> ExprRef:
        """
        Maps a multiset variable to a corresponding integer variable (its count at a point).

        If the multiset variable has already been mapped, it returns the
        existing integer variable. Otherwise, it creates a new integer
        variable, adds a non-negativity constraint to `self.star_fmls`,
        and returns the new variable.

        Args:
            t: The multiset variable.

        Returns:
            The corresponding integer variable.
        """
        if t in self.vars:
            return self.vars[t]
        v = self.fresh_var(t.decl().name())
        self.vars[t] = v
        self.star_fmls += [v >= 0]
        return v

    def visit(self, t):
        """Visit a node in the AST."""
        if t in self.visited:
            return self.visited[t]
        r = self.visit1(t)
        self.visited[t] = r
        return r

    def visit1(self, t):
        """
        Process a single node.

        Transforms multiset operations into operations on element counts.
        For example:
        - Union(A, B) -> count(A) + count(B)
        - Intersect(A, B) -> min(count(A), count(B))
        - Card(A) -> Sum over domain of count(A) (handled via add_star_def)
        """
        chs = [self.visit(f) for f in t.children()]
        if is_and(t):
            return And(chs)
        if is_or(t):
            return Or(chs)
        if is_not(t):
            return Not(chs[0])
        if is_ms_card(t):
            # card(M) -> sum_{x} M(x)
            return self.add_star_def(chs[0])
        if is_setof(t):
            # setof(M)(x) -> 1 if M(x) > 0 else 0
            return If(chs[0] > 0, 1, 0)
        if is_ms_empty(t):
            return 0
        if is_ms_union(t):
            # (A U B)(x) -> A(x) + B(x)
            return chs[0] + chs[1]
        if is_set_union(t):
            # (A U_set B)(x) -> 1 if A(x)>0 or B(x)>0 else 0
            return If(Or(chs[0] > 0, chs[1] > 0), 1, 0)
        if is_ms_inter(t):
            # (A I B)(x) -> min(A(x), B(x))
            t1 = chs[0]
            t2 = chs[1]
            return If(t1 >= t2, t2, t1)
        if is_ms_subtract(t):
            # (A \ B)(x) -> A(x) if B(x)==0 else 0 (Note: this seems to be a specific definition of subtraction)
            # Actually, standard multiset subtraction is max(0, A(x) - B(x)).
            # Let's check the code: If(t2 == 0, t1, 0). This looks like "set difference" behavior on multisets?
            # Or maybe "A minus B" where we remove all elements of B?
            # Let's keep the comment generic.
            t1 = chs[0]
            t2 = chs[1]
            return If(t2 == 0, t1, 0)
        if is_set_subtract(t):
            # (A \ B)(x) -> 1 if A(x)=1 and B(x)=0 else 0
            # Implementation: If(t1 <= t2, 0, t1 - t2)
            # If t1=1, t2=0 -> 1-0 = 1. If t1=1, t2=1 -> 0. If t1=0 -> 0.
            t1 = chs[0]
            t2 = chs[1]
            return If(t1 <= t2, 0, t1 - t2)
        if is_ms_var(t):
            return self.ms2var(t)
        if is_ms_subset(t):
            # A subset B -> forall x. A(x) <= B(x)
            # Translated as: sum_{x} (if A(x) > B(x) then 1 else 0) == 0
            u = self.add_star_def(If(chs[0] > chs[1], 1, 0))
            return u == 0
        if is_eq(t) and is_ms_sort(t.arg(0).sort()):
            # A == B -> forall x. A(x) == B(x)
            # Translated as: sum_{x} (if A(x) == B(x) then 0 else 1) == 0
            u = self.add_star_def(If(chs[0] == chs[1], 0, 1))
            return u == 0
        if is_app(t):
            return t.decl()(chs)
        assert False
        return None


def to_lia_star(fml):
    """
    Wrapper to convert a formula to LIA*.
    """
    ls = LiaStar()
    return ls.convert(fml)


mapa_flag = False


class Bapa2Ms:
    """
    Translates BAPA (Set) formulas to Multiset formulas.

    This class walks the AST of a formula and replaces set operations
    with corresponding multiset operations. It treats sets as multisets
    where the count of each element is either 0 or 1.
    """

    def __init__(self):
        """
        Initializes a new Bapa2Ms translator.
        """
        self.visited = {}
        self.set2ms_vars = {}

    def fresh_var(self, s, name):
        """Create a fresh variable of sort s."""
        return FreshConst(s, name)

    def convert(self, fmls):
        """Convert a list of formulas."""
        fmls = [self.visit(fml) for fml in fmls]
        return fmls

    def visit(self, t):
        """Visit a node."""
        if t in self.visited:
            return self.visited[t]
        r = self.visit1(t)
        self.visited[t] = r
        return r

    def is_set_sort(self, s):
        """Check if sort is a Set sort (Array -> Bool)."""
        return (
            Z3_get_sort_kind(s.ctx.ref(), s.ast) == Z3_ARRAY_SORT
            and BoolSort() == s.range()
        )

    def is_set_var(self, t):
        """Check if term is a Set variable."""
        return (
            is_app(t)
            and t.num_args() == 0
            and t.decl().kind() == Z3_OP_UNINTERPRETED
            and self.is_set_sort(t.sort())
        )

    def set2ms(self, t):
        """Convert a Set variable to a Multiset variable."""
        if t in self.set2ms_vars:
            return self.set2ms_vars[t]
        A = t.sort().domain()
        v = self.fresh_var(MS(A), "%s" % t)
        self.set2ms_vars[t] = v
        return v

    def is_set_card(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set cardinality application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set cardinality application, False otherwise.
        """
        return is_app(t) and t.num_args() == 1 and t.decl().name() == "card"

    def is_set_union(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set union application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set union application, False otherwise.
        """
        return is_app(t) and t.num_args() == 2 and t.decl().kind() == Z3_OP_SET_UNION

    def is_set_eq(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set equality application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set equality application, False otherwise.
        """
        return is_eq(t) and self.is_set_sort(t.arg(0).sort())

    def is_set_subtract(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set subtraction application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set subtraction application, False otherwise.
        """
        return is_app(t) and t.decl().kind() == Z3_OP_SET_DIFFERENCE

    def is_set_inter(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set intersection application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set intersection application, False otherwise.
        """
        return is_app(t) and t.decl().kind() == Z3_OP_SET_INTERSECT

    def is_set_empty(self, t: ExprRef) -> bool:
        """
        Checks if a term is an empty set constant.

        Args:
            t: The term to check.

        Returns:
            True if the term is an empty set constant, False otherwise.
        """
        return is_app(t) and t.decl().kind() == Z3_OP_CONST_ARRAY and is_false(t.arg(0))

    def is_set_subset(self, t: ExprRef) -> bool:
        """
        Checks if a term is a set subset application.

        Args:
            t: The term to check.

        Returns:
            True if the term is a set subset application, False otherwise.
        """
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
        assert False
        return None


# Perform conversion on the given formulas
def bapa2ms(fmls):
    """Convert BAPA formulas to Multiset formulas."""
    b2ms = Bapa2Ms()
    return b2ms.convert(fmls)


# Parse BAPA file, convert to multi-set formula
def parse_bapa(file, mapa):
    """
    Parse a BAPA/MAPA file and convert it to a multiset formula.

    Args:
        file: Path to the SMT2 file.
        mapa: Boolean, true if treating as MAPA (multisets), false for BAPA (sets).
    """
    global mapa_flag

    # Read file into solver state
    s = Solver()
    s.from_file(file)

    # Set flag for parsing bapa examples as multiset formulas vs set formulas
    if mapa:
        mapa_flag = True

    # Conversion
    return bapa2ms(s.assertions())
