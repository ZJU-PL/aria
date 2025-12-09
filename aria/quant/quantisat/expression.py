from itertools import product
from typing import List

import sympy as sp


def get_function_expression(name: str, parameters: List[sp.Symbol]) -> sp.Expr:
    """
    Get a uninterpreted function expression with the given the name and its list
    of parameters.

    Parameters
    ----------
    name : str
        The name of the function.
    parameters : List[sp.Symbol]
        The parameters of the function.

    Returns
    -------
    sp.Expr
        The function expression.
    """
    return sp.Function(name)(*parameters)


def get_polynomial_expression(coeffs_name: str, variables: List[sp.Symbol], degree: int) -> sp.Expr:
    """
    Get a polynomial expression over the given variables with the given degree.
    Coefficients are named with the given name and indexed for unique identification. 

    Parameters
    ----------
    coeffs_name : str
        The name of the coefficients.
    variables : List[sp.Symbol]
        The variables of the polynomial expression.
    degree : int
        The degree of the polynomial expression.

    Returns
    -------
    sp.Expr
        The polynomial expression.
    """
    monomials = get_all_monomials(variables, degree)
    coeffs = [sp.Symbol(f'{coeffs_name}_{i}') for i in range(len(monomials))]
    return sum([coeff * monomial for coeff, monomial in zip(coeffs, monomials)])


def get_all_monomials(variables: List[sp.Symbol], degree: int) -> List[sp.Expr]:
    """
    Get all monomials of a given degree over the variables.

    Parameters
    ----------
    variables : List[sp.Symbol]
        The variables of the monomials.
    degree : int
        The degree of the monomials.

    Returns
    -------
    List[sp.Expr]
        The list of monomials.
    """
    monomials = []
    for exponents in product(range(degree + 1), repeat=len(variables)):
        if sum(exponents) <= degree:
            monomials.append(sp.prod(
                [variable**exponent for variable, exponent in zip(variables, exponents)]))
    return monomials
