"""
- absolute_value_bv
- absolute_value_int
- get_variables
- get_atoms
- to_smtlib2
- is_function_symbol
- get_function_symbols
- skolemize
- big_and
- big_or
- negate
- is_expr_var
- is_expr_val
- is_term
- is_atom
- is_pos_lit
- is_neg_lit
- is_lit
- extract_all

- create_function_body_str
- z3_string_decoder
- z3_value_to_python
- get_z3_logic
"""

from typing import List, Set, Any
import z3


def absolute_value_bv(bv):
    """
    Based on: https://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
    Operation:
        Desired behavior is by definition (bv < 0) ? -bv : bv
        Now let mask := bv >> (bv.size() - 1)
        Note because of sign extension:
            bv >> (bv.size() - 1) == (bv < 0) ? -1 : 0
        Recall:
            -x == ~x + 1 => ~(x - 1) == -(x - 1) -1 == -x
            ~x == -1^x
             x ==  0^x
        now if bv < 0 then -bv == -1^(bv - 1) == mask ^ (bv + mask)
        else bv == 0^(bv + 0) == mask^(bv + mask)
        hence for all bv, absolute_value(bv) == mask ^ (bv + mask)
    """
    mask = bv >> (bv.size() - 1)
    return mask ^ (bv + mask)


def absolute_value_int(x):
    """
    Absolute value for integer encoding
    """
    return z3.If(x >= 0, x, -x)


def get_expr_vars(exp):
    """Get all variables from a Z3 expression."""
    try:
        syms = set()
        stack = [exp]

        while stack:
            e = stack.pop()
            if z3.is_app(e):
                if e.num_args() == 0 and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    syms.add(e)
                else:
                    # Add children in reverse order
                    # This maintains DFS traversal order when using pop()
                    # stack.extend(reversed(e.children())) # shoud we?
                    stack.extend(e.children())

        return list(syms)
    except z3.Z3Exception as ex:
        print(ex)
        return False


def get_expr_vars_v2(exp):
    """
    Get all variables from a Z3 expression.
    For complex expressions with shared subexpressions, add a visited set
    to avoid processing the same subexpression multiple times.
    """
    try:
        syms = set()
        visited = set()
        stack = [exp]

        while stack:
            e = stack.pop()
            e_id = z3.Z3_get_ast_id(e.ctx.ref(), e.as_ast())

            if e_id in visited:
                continue

            visited.add(e_id)

            if z3.is_app(e):
                if e.num_args() == 0 and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    syms.add(e)
                else:
                    stack.extend(e.children())

        return list(syms)
    except z3.Z3Exception as ex:
        print(ex)
        return False


def get_variables(exp: z3.ExprRef) -> [z3.ExprRef]:
    """Get variables of exp"""
    # return get_vars(exp)  # this can be very slow
    return get_expr_vars(exp)


def get_atoms(expr: z3.BoolRef):
    """
    Get all atomic predicates in a formula
    """
    s = set()

    def get_preds_(exp):
        if exp in s:
            return
        if z3.is_not(exp):
            s.add(exp)
        if z3.is_and(exp) or z3.is_or(exp):
            for e_ in exp.children():
                get_preds_(e_)
            return
        assert z3.is_bool(exp)
        s.add(exp)

    # convert to NNF and then look for preds
    ep = z3.Tactic("nnf")(expr).as_expr()
    get_preds_(ep)
    return s


def to_smtlib2(expr: z3.BoolRef) -> str:
    """ "
    To SMT-LIB2 string
    """
    s = z3.Solver()
    s.add(expr)
    return s.to_smt2()


def is_function_symbol(s: z3.ExprRef) -> bool:
    """Decide"""
    if not z3.is_app(s):
        return False
    if z3.is_const(s):
        return False

    func = s.decl()
    if func.range() == z3.BoolSort():
        # predicate symbol
        return False

    if func.name().lower() == "if":
        return False

    return True


def get_function_symbols(s: z3.ExprRef) -> Set[z3.FuncDeclRef]:
    """Find function symbols in a Z3 expr"""
    fsymbols = set()
    if is_function_symbol(s):
        fsymbols.add(s.decl())

    for child in s.children():
        fsymbols.update(get_function_symbols(child))

    return fsymbols


def skolemize(exp: z3.ExprRef) -> z3.ExprRef:
    """To Skolem normal form? (How about snf)"""
    g = z3.Goal()
    g.add(exp)
    t = z3.Tactic("snf")
    res = t(g)
    return res.as_expr()


def big_and(exp_list: List[z3.ExprRef]):
    """make and"""
    if len(exp_list) == 1:
        return exp_list[0]
    return z3.And(*exp_list)


def big_or(ll: List[z3.ExprRef]):
    """Make or."""
    if len(ll) == 1:
        return ll[0]
    return z3.Or(*ll)


def negate(f: z3.ExprRef) -> z3.ExprRef:
    """Negate a formula"""
    if z3.is_not(f):
        return f.arg(0)
    return z3.Not(f)


def ctx_simplify(exp: z3.ExprRef):
    """Perform complex simplifications (can be slow)"""
    return z3.Tactic("ctx-solver-simplify")(exp).as_expr()


def is_expr_var(a) -> bool:
    """
    Check if a is a variable. E.g. x is a var but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert is_expr_var(Int('7'))
    >>> assert not is_expr_var(IntVal('7'))
    >>> assert is_expr_var(Bool('y'))
    >>> assert not is_expr_var(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_var(LOnOff)
    >>> assert not is_expr_var(On)
    >>> assert is_expr_var(Block)
    >>> assert is_expr_var(SafetyInjection)
    """

    return z3.is_const(a) and a.decl().kind() == z3.Z3_OP_UNINTERPRETED


def is_expr_val(a) -> bool:
    """
    Check if the input formula is a value. E.g. 3 is a value but x = 3 is not.
    Examples:
    >>> from z3 import *
    >>> assert not is_expr_val(Int('7'))
    >>> assert is_expr_val(IntVal('7'))
    >>> assert not is_expr_val(Bool('y'))
    >>> assert not is_expr_val(Int('x') + 7 == Int('y'))
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> assert not is_expr_val(LOnOff)
    >>> assert is_expr_val(On)
    >>> assert not is_expr_val(Block)
    >>> assert not is_expr_val(SafetyInjection)
    """
    return z3.is_const(a) and a.decl().kind() != z3.Z3_OP_UNINTERPRETED


def is_term(a) -> bool:
    """
    Check if the input formula is a term. In FOL, terms are
    defined as term := const | var | f(t1,...,tn) where ti are terms.
    Examples:
    >>> from z3 import *
    >>> assert is_term(Bool('x'))
    >>> assert not is_term(And(Bool('x'),Bool('y')))
    >>> assert not is_term(And(Bool('x'),Not(Bool('y'))))
    >>> assert is_term(IntVal(3))
    >>> assert is_term(Int('x'))
    >>> assert is_term(Int('x') + Int('y'))
    >>> assert not is_term(Int('x') + Int('y') > 3)
    >>> assert not is_term(And(Int('x')==0,Int('y')==3))
    >>> assert not is_term(Int('x')==0)
    >>> assert not is_term(3)
    >>> assert not is_term(Bool('x') == (Int('y')==Int('z')))
    """

    if not z3.is_expr(a):
        return False
    if z3.is_const(a):  # covers both const value and var
        return True
    # covers f(t1,..,tn)
    return not z3.is_bool(a) and all(is_term(c) for c in a.children())


CONNECTIVE_OPS = [
    z3.Z3_OP_NOT,
    z3.Z3_OP_AND,
    z3.Z3_OP_OR,
    z3.Z3_OP_IMPLIES,
    z3.Z3_OP_IFF,
    z3.Z3_OP_ITE,
    z3.Z3_OP_XOR,
]


def is_atom(a) -> bool:
    """
    Check if the input formula is an atom. In FOL, atoms are
    defined as atom := t1 = t2 | R(t1,..,tn) where ti are terms.
    In addition, this function also allows Bool variable to
    be terms (in propositional logic, a bool variable is considered term)
    """
    if not z3.is_bool(a):
        return False

    if is_expr_val(a):
        return False

    if is_expr_var(a):
        return True

    return (
        z3.is_app(a)
        and a.decl().kind() not in CONNECTIVE_OPS
        and all(is_term(c) for c in a.children())
    )


def is_pos_lit(a) -> bool:
    """
    Check if the input formula is a positive literal,  i.e. an atom
    >>> is_pos_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_atom(a)


def is_neg_lit(a) -> bool:
    """
    Check if the input formula is a negative literal
    EXAMPLES:
    >>> from z3 import *
    >>> is_term(3)
    False
    >>> is_neg_lit(Not(Bool('x')))
    True
    >>> is_neg_lit(Not(BoolVal(False)))
    False
    >>> is_neg_lit(BoolVal(True))
    False
    >>> is_neg_lit(BoolVal(False))
    False
    >>> is_neg_lit(Not(Int('x') + Int('y') > 3))
    True
    >>> is_neg_lit(Not(Bool('x') == BoolVal(True)))
    True
    >>> is_neg_lit(Not(Int('x') == 3))
    True
    >>> is_neg_lit(Not(BoolVal(True)))
    False
    """
    return z3.is_not(a) and is_pos_lit(a.children()[0])


def is_lit(a) -> bool:
    """
    Check if the input formula is a negative literal
    >>> is_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_pos_lit(a) or is_neg_lit(a)


def create_function_body_str(funcname: str, varlist: List, body: z3.ExprRef) -> [str]:
    """Create SMT-LIB2 function definition string."""
    res = []
    target = f"(define-fun {funcname} ("
    for var in varlist:
        target += f"({var} {var.sort().sexpr()}) "
    target += f") Bool {body.sexpr()})"  # return value
    res.append(target)

    for var in varlist:
        res.append(f"(declare-const {var} {var.sort().sexpr()})")
    return res


def z3_string_decoder(z3str: z3.StringVal) -> str:
    """Convert z3.StringVal to python string"""
    length = z3.Int("length")
    tmp_string = z3.String("ts")
    solver = z3.Solver()
    solver.add(tmp_string == z3str)
    solver.add(z3.Length(tmp_string) == length)
    assert solver.check() == z3.sat

    model = solver.model()
    assert model[length].is_int()  # pylint: disable=no-member
    num_chars = model[length].as_long()  # pylint: disable=no-member

    solver.push()
    char_bvs = []
    for i in range(num_chars):
        char_bvs.append(z3.BitVec(f"ch_{i}", 8))
        solver.add(z3.Unit(char_bvs[i]) == z3.SubString(tmp_string, i, 1))

    assert solver.check() == z3.sat
    model = solver.model()
    python_string = "".join([chr(model[ch].as_long()) for ch in char_bvs])
    return python_string


def z3_value_to_python(value):
    """Convert a Z3 value to its Python equivalent."""
    if z3.is_true(value):
        return True
    if z3.is_false(value):
        return False
    if z3.is_int_value(value):
        return value.as_long()
    if z3.is_rational_value(value):
        return float(value.numerator_as_long()) / float(value.denominator_as_long())
    if z3.is_string_value(value):
        return z3_string_decoder(value)
    if z3.is_algebraic_value(value):
        raise NotImplementedError()
    raise NotImplementedError()


class FormulaInfo:
    """Information about a Z3 formula including quantifiers and logic."""

    def __init__(self, fml):
        self.formula = fml
        self.has_quantifier = self._check_has_quantifier()
        self.logic = self.get_logic()

    def apply_probe(self, name):
        """Apply a Z3 probe to the formula."""
        g = z3.Goal()
        g.add(self.formula)
        p = z3.Probe(name)
        return p(g)

    def _check_has_quantifier(self):
        """Check if the formula has quantifiers."""
        return self.apply_probe("has-quantifiers")

    def logic_has_bv(self):
        """Check if the logic contains bit-vectors."""
        return "BV" in self.logic

    def get_logic(self):
        """Determine the logic of the formula."""
        try:
            if not self.has_quantifier:
                if self.apply_probe("is-propositional"):
                    return "QF_UF"
                if self.apply_probe("is-qfbv"):
                    return "QF_BV"
                if self.apply_probe("is-qfaufbv"):
                    return "QF_AUFBV"
                if self.apply_probe("is-qflia"):
                    return "QF_LIA"
                # if self.apply_probe("is-quauflia"):
                #    return "QF_AUFLIA"
                if self.apply_probe("is-qflra"):
                    return "QF_LRA"
                if self.apply_probe("is-qflira"):
                    return "QF_LIRA"
                if self.apply_probe("is-qfnia"):
                    return "QF_NIA"
                if self.apply_probe("is-qfnra"):
                    return "QF_NRA"
                if self.apply_probe("is-qfufnra"):
                    return "QF_UFNRA"
                return "ALL"
            if self.apply_probe("is-lia"):
                return "LIA"
            if self.apply_probe("is-lra"):
                return "LRA"
            if self.apply_probe("is-lira"):
                return "LIRA"
            if self.apply_probe("is-nia"):
                return "NIA"
            if self.apply_probe("is-nra"):
                return "NRA"
            if self.apply_probe("is-nira"):
                return "NIRA"
            return "ALL"
        except (z3.Z3Exception, AttributeError) as ex:
            print(ex)
            return "ALL"


def get_z3_logic(fml: z3.ExprRef):
    """Get the logic of a Z3 formula."""
    fml_info = FormulaInfo(fml)
    return fml_info.get_logic()


def eval_predicates(m: z3.ModelRef, predicates: List[z3.BoolRef]):
    """Let m be a model of a formula phi, preds be a set of predicates"""
    res = []
    for p in predicates:
        if z3.is_true(m.eval(p)):
            res.append(p)
        elif z3.is_false(m.eval(p)):
            res.append(negate(p))
        else:
            pass
    return res


def extract_all(lst: List[Any]) -> List[Any]:
    """
    Extract all elements from nested lists.

    Args:
        lst: A potentially nested list

    Returns:
        A flattened list with all elements
    """
    results = []
    for elem in lst:
        if isinstance(elem, list):
            results.extend(extract_all(elem))
        else:
            results.append(elem)
    return results


if __name__ == "__main__":
    import doctest

    doctest.testmod()
