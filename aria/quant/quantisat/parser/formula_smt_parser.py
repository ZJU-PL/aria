from io import StringIO

import regex as re
import sympy as sp
from pysmt.smtlib.parser import SmtLibParser

from quantifier import Exists, ForAll, Quantifier


def parse_file(file_path):
    """
    Parse the formula from the given file.

    Parameters
    ----------
    file_path : str
        The path to the file to be parsed.

    Returns
    -------
    List[Formula]
        The parsed formula.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    parser = SmtLibParser()

    with StringIO(content) as stream:
        script = parser.get_script(stream)

    f = script.get_last_formula()

    if f.is_and():
        converted_system = [from_pysmt(sub_f) for sub_f in f.args()]
    else:
        converted_system = [from_pysmt(f)]

    all_free_symbols = set()
    for i, formula in enumerate(converted_system):
        free_symbols = formula.free_symbols
        all_free_symbols.update(free_symbols)

    for i, formula in enumerate(converted_system):
        if len(all_free_symbols) > 0:
            converted_system[i] = Exists(list(all_free_symbols), formula)

    return converted_system


def from_pysmt(expr) -> Quantifier:
    """
    Convert a pysmt expression to a quantified formula.

    Parameters
    ----------
    expr : pysmt.fnode.FNode
        The expression to convert.

    Returns
    -------
    Quantifier
        The quantified formula.
    """
    if expr.is_forall():
        assert len(
            expr.args()) == 1, f'Expected 1 argument, got {len(expr.args())}'
        return ForAll([sp.Symbol(var.symbol_name()) for var in expr.quantifier_vars()], from_pysmt(expr.arg(0)))
    elif expr.is_exists():
        assert len(
            expr.args()) == 1, f'Expected 1 argument, got {len(expr.args())}'
        return Exists([sp.Symbol(var.symbol_name()) for var in expr.quantifier_vars()], from_pysmt(expr.arg(0)))
    elif expr.is_le():
        assert len(
            expr.args()) == 2, f'Expected 2 arguments, got {len(expr.args())}'
        return sp.Le(from_pysmt(expr.arg(0)), from_pysmt(expr.arg(1)), evaluate=False)
    elif expr.is_lt():
        assert len(
            expr.args()) == 2, f'Expected 2 arguments, got {len(expr.args())}'
        return sp.Lt(from_pysmt(expr.arg(0)), from_pysmt(expr.arg(1)), evaluate=False)
    elif expr.is_equals():
        assert len(
            expr.args()) == 2, f'Expected 2 arguments, got {len(expr.args())}'
        return sp.Eq(from_pysmt(expr.arg(0)), from_pysmt(expr.arg(1)), evaluate=False)
    elif expr.is_implies():
        assert len(
            expr.args()) == 2, f'Expected 2 arguments, got {len(expr.args())}'
        return sp.Implies(from_pysmt(expr.arg(0)), from_pysmt(expr.arg(1)), evaluate=False)
    elif expr.is_and():
        assert len(
            expr.args()) >= 2, f'Expected more or equal to 2 arguments, got {len(expr.args())}'
        return sp.And(*[from_pysmt(arg) for arg in expr.args()], evaluate=False)
    elif expr.is_or():
        assert len(
            expr.args()) >= 2, f'Expected more or equal to 2 arguments, got {len(expr.args())}'
        return sp.Or(*[from_pysmt(arg) for arg in expr.args()], evaluate=False)
    elif expr.is_not():
        assert len(
            expr.args()) == 1, f'Expected 1 argument, got {len(expr.args())}'

        child = from_pysmt(expr.arg(0))
        if isinstance(child, Quantifier):
            return child.negate()
        else:
            return sp.Not(child, evaluate=False)
    elif expr.is_plus():
        assert len(
            expr.args()) >= 2, f'Expected more or equal to 2 arguments, got {len(expr.args())}'
        return sp.Add(*[from_pysmt(arg) for arg in expr.args()], evaluate=False)
    elif expr.is_minus():
        assert len(
            expr.args()) == 2, f'Expected 2 arguments, got {len(expr.args())}'
        return sp.Add(from_pysmt(expr.arg(0)), -from_pysmt(expr.arg(1)), evaluate=False)
    elif expr.is_times():
        assert len(
            expr.args()) >= 2, f'Expected more or equal to 2 arguments, got {len(expr.args())}'
        return sp.Mul(*[from_pysmt(arg) for arg in expr.args()], evaluate=False)
    elif expr.is_constant():
        assert len(
            expr.args()) == 0, f'Expected 0 arguments, got {len(expr.args())}'
        assert expr.constant_value() is not None, f'Expected a constant value, got None'
        # Check if the constant is an integer
        if re.match(r'^-?\d+$', str(expr.constant_value())):
            return sp.Number(int(expr.constant_value()))
        else:
            return sp.Number(float(expr.constant_value()))
    elif expr.is_symbol():
        assert len(
            expr.args()) == 0, f'Expected 0 arguments, got {len(expr.args())}'
        assert expr.symbol_name() is not None, f'Expected a symbol name, got None'
        return sp.Symbol(expr.symbol_name())
    else:
        raise ValueError(
            f'Unsupported expression type: {expr.node_type()} of expression {expr}')
