import signal
from enum import Enum
from typing import Callable
from warnings import warn

import sympy as sp

from quantifier import Exists, ForAll


class Result(Enum):
    """
    Enum for the result of the solver.
    """
    CORRECT = 1
    INCORRECT = 2
    CONVERSION_TIMEOUT = 3
    SOLVER_TIMEOUT = 4
    PARSING_ERROR = 5


def set_timeout(callable: Callable, timeout: int, *args, **kwargs) -> sp.Basic:
    """
    Set a timeout for a callable.

    Parameters
    ----------
    callable : Callable
        The callable to be executed.
    timeout : int
        The timeout in seconds.
    args : Any
        The arguments to be passed to the callable.
    kwargs : Any
        The keyword arguments to be passed to the callable.

    Returns
    -------
    Any
        The result of the callable.

    Raises
    ------
    TimeoutError
        If the operation times out.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(
            f'The operation {callable.__name__} timed out after {timeout} seconds.')

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        return callable(*args, **kwargs)
    finally:
        signal.alarm(0)


def to_smt(constraint: sp.Basic) -> str:
    """
    Convert a constraint to an SMT2 string.

    Parameters
    ----------
    constraint : sp.Basic (or Quantifier)
        The constraint to be converted.
    Returns
    -------
    str
        The SMT2 string representing the constraint.
    """
    if isinstance(constraint, Exists):
        return f'(exists ({" ".join([f"({to_smt(var)} Real)" for var in constraint.variables])}) {to_smt(constraint.formula)})'
    elif isinstance(constraint, ForAll):
        return f'(forall ({" ".join([f"({to_smt(var)} Real)" for var in constraint.variables])}) {to_smt(constraint.formula)})'
    elif constraint.is_Relational:
        assert len(
            constraint.args) == 2, f'Expected 2 arguments, got {len(constraint.args)}'
        arg_pair = f'{to_smt(constraint.lhs)} {to_smt(constraint.rhs)}'
        if constraint.rel_op == '==':
            # PolyHorn Bug
            return f'(and (<= {arg_pair}) (>= {arg_pair}))'
        elif constraint.rel_op == '!=':
            return f'(or (< {arg_pair}) (> {arg_pair}))'
        elif constraint.rel_op in ['<', '<=', '>', '>=']:
            return f'({constraint.rel_op} {arg_pair})'
        else:
            warn(f'Unsupported relational operator: {constraint.rel_op}')
            return f'({constraint.rel_op} {arg_pair})'
    elif constraint.is_Add:
        return f'(+ {" ".join([to_smt(arg) for arg in constraint.args])})'
    elif constraint.is_Mul:
        return f'(* {" ".join([to_smt(arg) for arg in constraint.args])})'
    elif constraint.is_Pow:
        return f'(* {" ".join([to_smt(constraint.base)] * int(constraint.exp))})'
    elif constraint.is_Function and constraint.is_Boolean:
        f = str(constraint.func).lower()
        if f == 'and':
            assert len(
                constraint.args) >= 2, f'Expected 2 arguments, got {len(constraint.args)}'
            return f'(and {" ".join([to_smt(arg) for arg in constraint.args])})'
        elif f == 'or':
            assert len(
                constraint.args) >= 2, f'Expected 2 arguments, got {len(constraint.args)}'
            return f'(or {" ".join([to_smt(arg) for arg in constraint.args])})'
        elif f == 'not':
            assert len(
                constraint.args) == 1, f'Expected 1 argument, got {len(constraint.args)}'
            child = constraint.args[0]
            if isinstance(child, sp.Implies):
                assert len(
                    child.args) == 2, f'Expected 2 arguments, got {len(child.args)}'
                return f'(and {to_smt(child.args[0])} {to_smt(sp.Not(child.args[1]))})'
            elif isinstance(child, sp.And):
                assert len(
                    child.args) >= 2, f'Expected 2 arguments, got {len(child.args)}'
                return f'(or {" ".join([to_smt(sp.Not(arg)) for arg in child.args])})'
            elif isinstance(child, sp.Or):
                assert len(
                    child.args) >= 2, f'Expected 2 arguments, got {len(child.args)}'
                return f'(and {" ".join([to_smt(sp.Not(arg)) for arg in child.args])})'
            else:
                raise ValueError(
                    f'Unable to reduce negation on: {type(child)}')
        elif f == 'implies':
            assert len(
                constraint.args) == 2, f'Expected 2 arguments, got {len(constraint.args)}'
            # return f'(=> {to_smt(constraint.args[0])} {to_smt(constraint.args[1])})'
            return f'(or {to_smt(sp.Not(constraint.args[0]))} {to_smt(constraint.args[1])})'
        else:
            warn(f'Unsupported function: {f}')
            return f'({str(constraint.func).lower()} {" ".join([to_smt(arg) for arg in constraint.args])})'
    elif constraint.is_Function:
        # Non-boolean functions (e.g. skolem functions)
        if len(constraint.args) == 0:
            return str(constraint.func)
        else:
            return f'({str(constraint.func)} {" ".join([to_smt(arg) for arg in constraint.args])})'
    elif isinstance(constraint, sp.UnevaluatedExpr):
        return to_smt(constraint.args[0])
    elif constraint.is_Symbol:
        return str(constraint)
    elif constraint.is_Number:
        return str(float(constraint))
    elif constraint == sp.true:
        return '(<= 0.0 1.0)'
    elif constraint == sp.false:
        return '(<= 1.0 0.0)'
    else:
        raise ValueError(
            f'Unsupported constraint type: {type(constraint)}\n\tFor constraint: {constraint}')


def split(a, n):
    """
    Split a list into n parts.

    Parameters
    ----------
    a : List
        The list to be split.
    n : int
        The number of parts to split the list.

    Returns
    -------
    List[List]
        The list of parts.
    """
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
