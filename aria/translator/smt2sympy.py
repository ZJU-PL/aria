# pylint: disable=all
"""
 This file is from https://github.com/ethanluoyc/sympais/blob/master/src/sympais/experimental/smt.py

For more translators from SMT-LIB2 to Computer Algebra System
- To ISAT, Mathematica, QepCAD, and Redlog
  https://github.com/cvc5/cvc5/tree/2283ee3b0327441c29caf26be977c1e4cd13c637/examples/nra-translate
- smt2sympy.py, osil2smt.py
  https://github.com/AndreaCallia/deltaopt
  https://github.com/AndreaCallia/manyopt
"""
import functools
import warnings
from typing import Callable, Dict
from pysmt.constants import is_pysmt_integer
from pysmt.shortcuts import get_env, reset_env
from pysmt.shortcuts import INT
from pysmt.shortcuts import REAL
from pysmt.smtlib.parser import SmtLibParser
from pysmt.walkers import DagWalker
from pysmt.walkers import IdentityDagWalker
from six.moves import cStringIO
import sympy

default_interpreted_constants = {'PI': sympy.pi, 'E': sympy.exp(1)}

# Additional sympy functions can be added similarly:
# https://docs.sympy.org/latest/modules/functions/index.html
default_interpreted_unary_functions = {
    'sin': sympy.sin,
    'cos': sympy.cos,
    'tan': sympy.tan,
    'asin': sympy.asin,
    'acos': sympy.acos,
    'atan': sympy.atan,
    'log': sympy.log,
    'exp': sympy.exp,
    'sqrt': sympy.sqrt,
}


def _make_walk_nary(func):
    def walk_nary(self, formula, args, **kwargs):
        return func(args)

    return walk_nary


def _make_walk_binary(func):
    def walk_binary(self, formula, args, **kwargs):
        assert len(args) == 2
        return func(args)

    return walk_binary


class SMTToSympyWalker(IdentityDagWalker):

    def __init__(self, environment, interpreted_constants: Dict[str, Callable],
                 interpreted_unary_functions: Dict[str, Callable]):
        DagWalker.__init__(self, environment)
        self.mgr = environment.formula_manager
        self._get_type = environment.stc.get_type
        self._back_memoization = {}
        self._interpreted_constants = interpreted_constants
        self._interpreted_unary_functions = interpreted_unary_functions

    def walk_true(self, formula, args, **kwargs):
        return True

    def walk_false(self, formula, args, **kwargs):
        return False

    def walk_symbol(self, formula, args, **kwargs):
        symbol_type = formula.symbol_type()
        symbol_name = formula.symbol_name()
        if symbol_type == REAL:
            return self._interpreted_constants.get(symbol_name,
                                                   sympy.Symbol(symbol_name, real=True))
        if symbol_type == INT:
            return self._interpreted_constants.get(symbol_name,
                                                   sympy.Symbol(symbol_name, integer=True))
        assert (symbol_type != REAL and
                symbol_type != INT), (f'Only real and integer variables are allowed: '
                                      f'{symbol_name} has type {symbol_type}')

    def walk_ite(self, formula, args, **kwargs):
        # ITE(condition, A, B)
        return sympy.ITE(args[0], args[1], args[2])

    def walk_real_constant(self, formula, **kwargs):
        frac = formula.constant_value()
        n, d = frac.numerator, frac.denominator
        return sympy.Rational(n, d)

    def walk_int_constant(self, formula, **kwargs):
        assert is_pysmt_integer(formula.constant_value())
        return int(str(formula.constant_value()))

    def walk_bool_constant(self, formula, **kwargs):
        raise NotImplementedError()

    def walk_quantifier(self, formula, args, **kwargs):
        raise NotImplementedError()

    def walk_toreal(self, formula, args, **kwargs):
        return sympy.Rational(args[0])

    def walk_realtoint(self, formula, args, **kwargs):
        del formula
        return int(args[0])

    def walk_function(self, formula, args, **kwargs):
        function_name = formula.function_name().symbol_name()
        if function_name in self._interpreted_unary_functions:
            assert len(args) == 1
            return self._interpreted_unary_functions[function_name](*args)
        raise NotImplementedError()

    walk_and = _make_walk_nary(
        lambda args: functools.reduce(lambda x, y: x & y, args, True))
    walk_or = _make_walk_nary(
        lambda args: functools.reduce(lambda x, y: x | y, args, False))
    walk_not = lambda args: sympy.Not(args[0])

    walk_plus = _make_walk_nary(
        lambda args: functools.reduce(lambda x, y: x + y, args, 0))
    walk_times = _make_walk_nary(lambda args: functools.reduce(lambda x, y: x * y, args))
    walk_minus = _make_walk_nary(lambda args: functools.reduce(lambda x, y: x - y, args))

    walk_implies = _make_walk_binary(lambda args: sympy.Implies(args[0], args[1]))
    walk_iff = _make_walk_binary(
        lambda args: (sympy.Implies(args[0], args[1]) & sympy.Implies(args[1], args[0])))
    walk_pow = _make_walk_binary(lambda args: args[0] ** args[1])
    walk_div = _make_walk_binary(lambda args: args[0] / args[1])

    walk_equals = _make_walk_binary(lambda args: sympy.Eq(args[0], args[1]))
    walk_le = _make_walk_binary(lambda args: args[0] <= args[1])
    walk_lt = _make_walk_binary(lambda args: args[0] < args[1])

    walk_exists = walk_quantifier
    walk_forall = walk_quantifier


def smtlib_to_sympy_constraint(
        smtlib_input: str,
        interpreted_constants: Dict[str, Callable] = default_interpreted_constants,
        interpreted_unary_functions: Dict[str,
        Callable] = default_interpreted_unary_functions):
    """Convert SMTLIB(v2) constraints into sympy constraints analyzable via SYMPAIS.
    This function is experimental and introduced as an example.
    It is implemented on top of PySMT (https://github.com/pysmt/pysmt).
    Additional features can be added extending the `SMTToSympyWalker` class.
    Args:
      smtlib_input: SMT constraint as a string in SMTLIB(v2) format, as accepted by PySMT
      interpreted_constants:
        predefined interpreted constants to be declared in the SMT problem.
        Default: E (Euler), PI
      interpreted_unary_functions:
        predefined interpreted functions Real -> Real.
        Default: sin, cos, tan, asin, acos, atan, log, exp, sqrt
    Returns:
      A dict of the estimates found by the DMC sampler.
    """

    interpreted_symbols_declarations = '\n'.join(
        [f'(declare-const {cname} Real)' for cname in interpreted_constants.keys()])
    interpreted_symbols_declarations += '\n'.join([
        f'(declare-fun {fname} (Real) Real)'
        for fname in interpreted_unary_functions.keys()
    ])

    smtlib_with_interpreted_symbols = (
            interpreted_symbols_declarations + '\n' + smtlib_input)

    reset_env()
    parser = SmtLibParser()
    script = parser.get_script(cStringIO(smtlib_with_interpreted_symbols))
    f = script.get_last_formula()
    converter = SMTToSympyWalker(get_env(), interpreted_constants,
                                 interpreted_unary_functions)
    f_sympy = converter.walk(f)
    f_sympy = sympy.logic.simplify_logic(f_sympy)
    f_sympy = sympy.simplify(f_sympy)
    if f_sympy.atoms(sympy.logic.Or):
        warnings.warn(
            'Disjunctive constraints are not supported by RealPaver. Consider replacing it with an adequate interval constraint propagation tool for benefit from all the features of SYMPAIS'
        )
    return f_sympy


def test_example_smt_transformation():
    smt01 = '''
    (declare-fun x () Real)
    (declare-fun y () Real)
    (declare-const PI Real)
    (assert
    (<=
        (+
            (* (sin x) (sin (/ PI 2.0)))
            (* y (sqrt y))
        )
        1
        )
    )
    (assert
        (> x 20)
    )
    '''
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    expected_result = (y * sympy.sqrt(y) + sympy.sin(x) <= 1) & (x > 20)
    print(expected_result)
    assert sympy.logic.boolalg.Equivalent(expected_result,
                                          smtlib_to_sympy_constraint(smt01))


if __name__ == '__main__':
    test_example_smt_transformation()
