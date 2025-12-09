"""
Some APIs/functions for playing with Z3 exper

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

- create_function_body_str
- z3_string_decoder
- z3_value_to_python
- get_z3_logic
"""

"""
Utility functions for working with Z3 expressions, formulas, and solvers.

This module provides helper functions for analyzing Z3 expressions, manipulating
formulas, and extracting information from Z3 objects.

Functions:
---------
Formula analysis:
- get_variables: Extract variables from a Z3 expression
- get_atoms: Extract atomic predicates from a Z3 formula
- get_function_symbols: Find function symbols in a Z3 expression
- get_z3_logic: Determine the SMT-LIB2 logic fragment for a Z3 expression

Formula classification:
- is_expr_var: Check if expression is a variable
- is_expr_val: Check if expression is a value
- is_term: Check if expression is a first-order logic term
- is_atom: Check if expression is an atom
- is_pos_lit: Check if expression is a positive literal
- is_neg_lit: Check if expression is a negative literal
- is_lit: Check if expression is a literal
- is_function_symbol: Check if expression is a function symbol

Sort checking:
- is_bool_sort: Check if expression has Boolean sort
- is_bv_sort: Check if expression has bit-vector sort
- is_int_sort: Check if expression has integer sort
- is_real_sort: Check if expression has real sort

Formula manipulation:
- negate: Negate a formula
- big_and: Create conjunction from a list of formulas
- big_or: Create disjunction from a list of formulas
- skolemize: Convert formula to Skolem normal form
- ctx_simplify: Perform context-dependent formula simplification

Conversion and serialization:
- to_smtlib2: Convert formula to SMT-LIB2 string format
- create_function_body_str: Create SMT-LIB2 function definition
- z3_string_decoder: Convert Z3 StringVal to Python string
- z3_value_to_python: Convert Z3 values to Python objects

Classes:
-------
- FormulaInfo: Analyze and extract information from Z3 formulas
"""

from typing import List, Set, Union, Tuple, Iterator
import z3
from z3.z3util import get_vars


def get_expr_vars(exp) -> List[z3.ExprRef]:
    """z3.z3util.get_vars can be very slow; so we use the
    cutomized version.
    TBD: check if this is correct and design a more efficient version.
    """
    try:
        syms = set()
        stack = [exp]

        while stack:
            e = stack.pop()
            if z3.is_app(e):
                if e.num_args() == 0 and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                    syms.add(e)
                else:
                    stack.extend(e.children())

        return list(syms)
    except z3.Z3Exception as ex:
        print(ex)
        return False


def get_variables(exp: z3.ExprRef) -> List[z3.ExprRef]:
    """Get variables of exp
    Examples:
    >>> from z3 import *
    >>> x, y = Ints('x y')
    >>> f = x + y
    >>> get_expr_vars(f)
    [x, y]
    """
    # return get_vars(exp)  # this can be very slow
    return get_expr_vars(exp)


def get_atoms(expr: z3.BoolRef) -> Set[z3.BoolRef]:
    """Get all atomic predicates in a Z3 formula.

    This function extracts all atomic boolean predicates from a Z3 formula by first converting it
    to Negation Normal Form (NNF) and then recursively traversing the formula structure.

    Args:
        expr (z3.BoolRef): A Z3 boolean formula

    Returns:
        Set[z3.BoolRef]: A set containing all atomic predicates found in the formula

    Examples:
        >>> import z3
        >>> x, y = z3.Ints('x y')
        >>> f = z3.And(x > 1, z3.Or(y < 0, z3.Not(x == 2)))
        >>> atoms = get_atoms(f)
        >>> len(atoms)
        3
        >>> print(atoms)
        {x > 1, Not(x == 2), y < 0}

    Notes:
        - The function first converts the input formula to NNF using Z3's 'nnf' tactic
        - Atomic predicates include arithmetic comparisons and their negations
        - The function handles AND/OR operations by recursively processing their children
    """
    a_set = set()

    def get_preds_(exp):
        if exp in a_set:
            return
        if z3.is_not(exp):
            a_set.add(exp)
        if z3.is_and(exp) or z3.is_or(exp):
            for e_ in exp.children():
                get_preds_(e_)
            return
        assert (z3.is_bool(exp))
        a_set.add(exp)

    # convert to NNF and then look for preds
    exp = z3.Tactic('nnf')(expr).as_expr()
    # exp = z3.Then('simplify', 'nnf')(expr).as_expr() # shoud we do this?
    get_preds_(exp)
    return a_set


def to_smtlib2(expr: z3.BoolRef) -> str:
    """"
    To SMT-LIB2 string
    """
    sol = z3.Solver()
    sol.add(expr)
    return sol.to_smt2()


def is_function_symbol(exp: z3.ExprRef) -> bool:
    """Decide if exp is a function symbol"""
    if not z3.is_app(exp):
        return False
    if z3.is_const(exp):
        return False

    func = exp.decl()
    if func.range() == z3.BoolSort():
        # predicate symbol
        return False

    if func.name().lower() == 'if':
        return False

    return True


def get_function_symbols(exp: z3.ExprRef) -> Set[z3.FuncDeclRef]:
    """Find function symbols in a Z3 expr"""
    fsymbols = set()
    if is_function_symbol(exp):
        fsymbols.add(exp.decl())

    for child in exp.children():
        fsymbols.update(get_function_symbols(child))

    return fsymbols


def skolemize(exp: z3.ExprRef) -> z3.ExprRef:
    """To Skolem normal form?
    About Skolemization: https://en.wikipedia.org/wiki/Skolemization
    In short, Skolemization is a process of removing existential quantifiers from a formula.
    Rules:
    - ∃x. φ(x) → φ(c) where c is a new Skolem constant.
    - ∀x. ∃y. φ(x,y) → ∀x. φ(x,f(x)) where f is a new Skolem function.
    In the second rule, f is a new function introduced for the existential quantifier
    """
    goal = z3.Goal()
    goal.add(exp)
    tactic = z3.Tactic('snf')
    res = tactic(goal)
    return res.as_expr()


def z3_quantifier_alternations(e: z3.ExprRef) -> Iterator[Tuple[z3.SortRef, z3.SortRef]]:
    """
    Analyze quantifier alternations in a Z3 expression by examining Skolem functions.

    This function identifies quantifier alternations by:
    1. Skolemizing the input expression (removing existential quantifiers)
    2. Finding all function symbols introduced during skolemization
    3. For each Skolem function, yielding (domain_sort, range_sort) pairs for each argument

    The Skolem functions represent existential quantifiers that depend on universal quantifiers,
    indicating quantifier alternations in the original formula.

    Args:
        e: A Z3 expression to analyze for quantifier alternations

    Yields:
        Tuple[z3.SortRef, z3.SortRef]: Pairs of (domain_sort, range_sort) for each
        argument position of each Skolem function found in the skolemized expression.
        Each pair represents a quantifier alternation where the domain sort corresponds
        to the universal quantifier and the range sort corresponds to the existential quantifier.

    Example:
        For formula ∀x. ∃y. P(x, y), skolemization introduces a function f(x) where
        f has domain sort matching x and range sort matching y, indicating a
        ∀∃ alternation.
    """
    skolemized = skolemize(e)
    for fsym in get_function_symbols(skolemized):
        for i in range(0, fsym.arity()):
            yield (fsym.domain(i), fsym.range())


def big_and(exp_list: List[z3.ExprRef]):
    """Conjunction of a list of expressions"""
    if len(exp_list) == 1:
        return exp_list[0]
    return z3.And(*exp_list)


def big_or(list_of_exp: List[z3.ExprRef]):
    """Disjunction of a list of expressions"""
    if len(list_of_exp) == 1:
        return list_of_exp[0]
    return z3.Or(*list_of_exp)


def negate(fml: z3.ExprRef) -> z3.ExprRef:
    """Negate a formula"""
    if z3.is_not(fml):
        return fml.arg(0)
    return z3.Not(fml)


def ctx_simplify(exp: z3.ExprRef):
    """Perform complex simplifications (can be slow)"""
    return z3.Tactic('ctx-solver-simplify')(exp).as_expr()


def is_expr_var(exp) -> bool:
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

    return z3.is_const(exp) and exp.decl().kind() == z3.Z3_OP_UNINTERPRETED


def is_expr_val(exp) -> bool:
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
    return z3.is_const(exp) and exp.decl().kind() != z3.Z3_OP_UNINTERPRETED


def is_term(exp) -> bool:
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

    if not z3.is_expr(exp):
        return False
    if z3.is_const(exp):  # covers both const value and var
        return True
    return not z3.is_bool(exp) and all(is_term(c) for c in exp.children())


CONNECTIVE_OPS = [z3.Z3_OP_NOT, z3.Z3_OP_AND, z3.Z3_OP_OR, z3.Z3_OP_IMPLIES,
                  z3.Z3_OP_IFF, z3.Z3_OP_ITE, z3.Z3_OP_XOR]


def is_atom(exp) -> bool:
    """
    Check if the input formula is an atom. In FOL, atoms are
    defined as atom := t1 = t2 | R(t1,..,tn) where ti are terms.
    In addition, this function also allows Bool variable to
    be terms (in propositional logic, a bool variable is considered term)
    """
    if not z3.is_bool(exp):
        return False

    if is_expr_val(exp):
        return False

    if is_expr_var(exp):
        return True

    return z3.is_app(exp) and exp.decl().kind() not in CONNECTIVE_OPS and \
        all(is_term(c) for c in exp.children())


def is_pos_lit(fml) -> bool:
    """
    Check if the input formula is a positive literal,  i.e. an atom
    >>> is_pos_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_atom(fml)


def is_neg_lit(exp) -> bool:
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
    return z3.is_not(exp) and is_pos_lit(exp.children()[0])


def is_lit(exp) -> bool:
    """
    Check if the input formula is a negative literal
    >>> is_lit(z3.Not(z3.BoolVal(True)))
    False
    """
    return is_pos_lit(exp) or is_neg_lit(exp)


def create_function_body_str(funcname: str, varlist: List, body: z3.ExprRef) -> [str]:
    """
    Creates a string representation of a function body which can be used to define a new function in SMT-LIB2 format.

    Parameters:
    -----------
    funcname : str
        The name of the function to be defined.
    varlist : List
        The list of input variables for the function.
    body : z3.ExprRef
        The body of the function as a Z3 expression.

    Returns:
    -------
    str
        A string representation of the function body in SMT-LIB2 format.
    """
    res = []
    target = "(define-fun {} (".format(funcname)
    for i in range(len(varlist)):
        target += "({} {}) ".format(str(varlist[i]), varlist[i].sort().sexpr())
    target += ") Bool {})".format(body.sexpr())  # return value
    res.append(target)

    for var in varlist:
        res.append("(declare-const {} {})".format(var, var.sort().sexpr()))
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
    assert model[length].is_int()
    num_chars = model[length].as_long()

    solver.push()
    char_bvs = []
    for i in range(num_chars):
        char_bvs.append(z3.BitVec("ch_%d" % i, 8))
        solver.add(z3.Unit(char_bvs[i]) == z3.SubString(tmp_string, i, 1))

    assert solver.check() == z3.sat
    model = solver.model()
    python_string = "".join([chr(model[ch].as_long()) for ch in char_bvs])
    return python_string


def z3_value_to_python(value) -> any:
    """Convert a Z3 value to a python value"""
    if z3.is_true(value):
        return True
    elif z3.is_false(value):
        return False
    elif z3.is_int_value(value):
        return value.as_long()
    elif z3.is_rational_value(value):
        return float(value.numerator_as_long()) / float(value.denominator_as_long())
    elif z3.is_string_value(value):
        return z3_string_decoder(value)
    elif z3.is_algebraic_value(value):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


class FormulaInfo:
    """For formula info
    Examples:
    >>> from z3 import *
    >>> x, y = Ints('x y')
    >>> f = x + y
    >>> fml_info = FormulaInfo(f)
    >>> fml_info.get_logic()
    'LIA'
    """

    def __init__(self, fml):
        self.formula = fml
        self._has_quantifier = None  # Initialize as None
        self._logic = None  # Initialize as None

    def apply_probe(self, name):
        goal = z3.Goal()
        goal.add(self.formula)
        probe = z3.Probe(name)
        return probe(goal)

    def has_quantifier(self):
        if self._has_quantifier is None:  # Only compute if not cached
            self._has_quantifier = self.apply_probe('has-quantifiers')
        return self._has_quantifier

    def logic_has_bv(self):
        return "BV" in self.get_logic()

    def has_theory(self, theory_name):
        """Check if formula contains a specific theory.

        Args:
            theory_name (str): Name of the theory to check for ('array', 'fp', 'string', etc.)

        Returns:
            bool: True if the formula contains the specified theory
        """
        try:
            if theory_name.lower() == 'array':
                return self.apply_probe('has-arrays')
            elif theory_name.lower() == 'fp':
                return self.apply_probe('has-fp')
            elif theory_name.lower() == 'string':
                # Probe for string-related functions or constants
                result = False
                goal = z3.Goal()
                goal.add(self.formula)
                string_indicators = ['str.', 'seq.', 'string']
                sexpr = goal.sexpr()
                return any(indicator in sexpr for indicator in string_indicators)
            elif theory_name.lower() == 'bv':
                return self.apply_probe('has-bit2bool') or self.apply_probe('is-qfbv')
            else:
                return False
        except Exception:
            return False

    def get_logic(self):
        """Determine the SMT-LIB2 logic for the formula.

        This method identifies the appropriate SMT-LIB2 logic based on the formula's
        characteristics by analyzing theories present (arithmetic, bit-vectors, arrays,
        strings, floating-point) and the presence of quantifiers.

        The method uses Z3 probes to detect formula properties and builds the logic
        string according to SMT-LIB2 naming conventions:
        - QF_* for quantifier-free formulas
        - *LIA/*LRA for linear integer/real arithmetic
        - *NIA/*NRA for non-linear integer/real arithmetic
        - *BV for bit-vectors
        - *A for arrays
        - *S for strings
        - *FP for floating-point
        - *UF for uninterpreted functions

        Returns:
            str: The SMT-LIB2 logic string that best describes the formula

        Examples:
            >>> x, y = z3.Ints('x y')
            >>> f = z3.And(x > 0, y < 10)
            >>> FormulaInfo(f).get_logic()
            'QF_LIA'

            >>> a = z3.Array('a', z3.IntSort(), z3.IntSort())
            >>> g = a[x] == y
            >>> FormulaInfo(g).get_logic()
            'QF_ALIA'
        """
        if self._logic is not None:  # Return cached value if available
            return self._logic

        try:
            # Detect theories present in the formula
            has_arrays = self.has_theory('array')
            has_fp = self.has_theory('fp')
            has_strings = self.has_theory('string')
            has_bv = self.has_theory('bv')

            # Base logic string
            logic_str = ""
            if not self.has_quantifier():
                logic_str = "QF_"

            # Detect arithmetic theories
            is_arith = False
            arith_type = ""

            if not self.has_quantifier():  # Quantifier-free logic
                if self.apply_probe("is-qflia"):
                    arith_type = "LIA"
                    is_arith = True
                elif self.apply_probe("is-qflra"):
                    arith_type = "LRA"
                    is_arith = True
                elif self.apply_probe("is-qfnia"):
                    arith_type = "NIA"
                    is_arith = True
                elif self.apply_probe("is-qfnra"):
                    arith_type = "NRA"
                    is_arith = True
                elif self.apply_probe("is-qflira"):
                    arith_type = "LIRA"
                    is_arith = True
            else:  # Quantified logic
                if self.apply_probe("is-lia"):
                    arith_type = "LIA"
                    is_arith = True
                elif self.apply_probe("is-lra"):
                    arith_type = "LRA"
                    is_arith = True
                elif self.apply_probe("is-nia"):
                    arith_type = "NIA"
                    is_arith = True
                elif self.apply_probe("is-nra"):
                    arith_type = "NRA"
                    is_arith = True
                elif self.apply_probe("is-nira") or self.apply_probe("is-lira"):
                    arith_type = "LIRA"
                    is_arith = True

            # Build the logic string based on all theories present
            theories = []

            # Add array support if present
            if has_arrays:
                theories.append("A")

            # Add bit-vector support if present
            if has_bv:
                theories.append("BV")

            # Add floating-point support if present
            if has_fp:
                theories.append("FP")

            # Add string support if present
            if has_strings:
                theories.append("S")

            # Check for uninterpreted functions
            if self.apply_probe("has-uninterpreted-functions") or self.apply_probe("is-propositional"):
                theories.append("UF")

            # Combine theories with the arithmetic type
            if is_arith:
                theories.append(arith_type)

            if not theories:
                if self.apply_probe("is-propositional"):
                    self._logic = "QF_UF"
                else:
                    self._logic = "ALL"
            else:
                # Sort the theories alphabetically except arithmetic which comes last
                non_arith = sorted([t for t in theories if t != arith_type])
                if arith_type:
                    theories = non_arith + [arith_type]
                else:
                    theories = non_arith

                self._logic = logic_str + "".join(theories)

            return self._logic

        except Exception as ex:
            import logging
            logging.warning(f"Error determining SMT logic: {ex}")
            self._logic = "ALL"
            return self._logic


def get_z3_logic(fml: z3.ExprRef) -> str:
    """Determine the SMT-LIB2 logic fragment that best describes a Z3 expression.

    This function is a convenient wrapper around FormulaInfo.get_logic() that takes a Z3
    expression and returns the appropriate SMT-LIB2 logic name.

    The logic name follows SMT-LIB2 conventions:
    - Prefix QF_ for quantifier-free formulas
    - Suffix letters for theories:
      - LIA/LRA: Linear integer/real arithmetic
      - NIA/NRA: Non-linear integer/real arithmetic
      - BV: Bit-vectors
      - A: Arrays
      - S: Strings
      - FP: Floating-point
      - UF: Uninterpreted functions

    Args:
        fml (z3.ExprRef): A Z3 expression

    Returns:
        str: SMT-LIB2 logic name that best describes the expression

    Examples:
        >>> import z3
        >>> x, y = z3.Ints('x y')
        >>> get_z3_logic(x + y > 0)
        'QF_LIA'
        >>> a = z3.Array('a', z3.IntSort(), z3.RealSort())
        >>> get_z3_logic(a[x] > 0.0)
        'QF_ALRA'
    """
    try:
        fml_info = FormulaInfo(fml)
        return fml_info.get_logic()
    except Exception as ex:
        import logging
        logging.warning(f"Error determining logic for expression: {ex}")
        return "ALL"  # Default to most general logic if detection fails


def is_bool_sort(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a Boolean variable.

    Args:
        expr: The Z3 expression to check

    Returns:
        True if the expression is a Boolean variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.BoolSort()


def is_bv_sort(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a bit-vector variable.

    Args:
        expr: The Z3 expression to check

    Returns:
        True if the expression is a bit-vector variable, False otherwise
    """
    return z3.is_const(expr) and z3.is_bv_sort(expr.sort())


def is_int_sort(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is an integer variable.

    Args:
        expr: The Z3 expression to check

    Returns:
        True if the expression is an integer variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.IntSort()


def is_real_sort(expr: z3.ExprRef) -> bool:
    """
    Check if a Z3 expression is a real variable.

    Args:
        expr: The Z3 expression to check

    Returns:
        True if the expression is a real variable, False otherwise
    """
    return z3.is_const(expr) and expr.sort() == z3.RealSort()


def eval_predicates(model: z3.ModelRef, predicates: List[z3.BoolRef]) -> List[z3.BoolRef]:
    """
    Let m be a model of a formula phi, preds be a set of predicates
    """
    res = []
    for pred in predicates:
        if z3.is_true(model.eval(pred)):
            res.append(pred)
        elif z3.is_false(model.eval(pred)):
            res.append(negate(pred))
        else:
            pass
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()

if __name__ == "__main__":
    import doctest


    # Additional test cases for new get_z3_logic functionality
    def additional_tests():
        """
        >>> import z3
        >>> x, y = z3.Ints('x y')
        >>> a, b = z3.Reals('a b')
        >>> bv1, bv2 = z3.BitVecs('bv1 bv2', 8)
        >>> s1, s2 = z3.Strings('s1 s2')
        >>> fp1, fp2 = z3.FPs('fp1 fp2', z3.FPSort(8, 24))

        # Test integer arithmetic
        >>> get_z3_logic(z3.And(x > 0, y < 10))
        'QF_LIA'

        # Test real arithmetic
        >>> get_z3_logic(z3.And(a > 0.0, b < 10.0))
        'QF_LRA'

        # Test mixed arithmetic
        >>> get_z3_logic(z3.And(x > 0, a < 10.0))
        'QF_LIRA'

        # Test bit-vectors
        >>> get_z3_logic(bv1 + bv2 == 0)
        'QF_BV'

        # Test arrays
        >>> arr = z3.Array('arr', z3.IntSort(), z3.IntSort())
        >>> get_z3_logic(arr[x] == y)
        'QF_ALIA'

        # Test bit-vector arrays
        >>> bv_arr = z3.Array('bv_arr', z3.BitVecSort(8), z3.BitVecSort(8))
        >>> get_z3_logic(bv_arr[bv1] == bv2)
        'QF_ABV'

        # Test strings
        >>> get_z3_logic(s1 == s2)
        'QF_S'

        # Test floating point
        >>> get_z3_logic(fp1 > fp2)
        'QF_FP'

        # Test quantifiers
        >>> get_z3_logic(z3.ForAll([x], x > 0))
        'LIA'

        # Test combination of theories
        >>> get_z3_logic(z3.And(arr[x] == y, s1 == s2))
        'QF_ALIAS'
        """
        pass


    doctest.testmod()
