# This module defines some general utilities that are common to many modules in the naturalproofs package.

import z3


# Hack to get a FuncDeclRef corresponding to Implies and IsSubset
Implies_as_FuncDeclRef = z3.Implies(True, True).decl()
IsSubset_Int_as_FuncDeclRef = z3.IsSubset(z3.EmptySet(z3.IntSort()), z3.EmptySet(z3.IntSort())).decl()


def apply_bound_formula(bound_formula, args):
    """
    A bound formula defines a macro formal_params -> body where formal_params are the formal parameters for the macro
    and are represented as a tuple of variables. The body is a z3.ExprRef that is defined in terms of these formal
    parameters. This function 'applies' the macro on the given args. bound_formula is the pair (formal_params, body). If
    the macro takes no parameters, then formal_params is ().  
    :param bound_formula: (tuple of z3.ExprRef, z3.ExprRef)  
    :param args: tuple of z3.ExprRef  
    :return: z3.ExprRef  
    """
    formal_params, body = bound_formula
    arity = len(formal_params)
    if len(args) != arity:
        raise TypeError('The arities of the bound formula and the arguments do not match.')
    return z3.substitute(body, [(formal_params[i], args[i]) for i in range(arity)])


def transform_expression(expression, transformations):
    """
    Apply one or more 'transformations' to an expression. Each transformation is a pair (cond, op). The transformation
    op is applied if cond is satisfied by the expression. The transformations are applied bottom-up on the parse tree of
    the expression, with the first matching transformation being the one applied. If no transformations are applicable,
    the original expression is returned.  
    This function acts as a generalisation of z3.substitute  
    :param expression: z3.ExprRef  
    :param transformations: list of (function: z3.ExprRef -> bool, function z3.ExprRef -> z3.ExprRef)  
    :return: z3.ExprRef  
    """
    declaration = expression.decl()
    args = expression.children()
    transformed_args = tuple([transform_expression(arg, transformations) for arg in args])
    transformed_expr_rec = None
    try:
        transformed_expr_rec = declaration(*transformed_args)
        # More expensive but reliable method using substitute. Commented out for now.
        # transformed_expr_rec = z3.substitute(expression, list(zip(args, transformed_args)))
    except Exception:
        exit('Something has gone wrong with formula substitution.')
    if isinstance(transformations, tuple):
        # Only one transformation given
        transformations = [transformations]
    for transformation in transformations:
        cond, op = transformation
        try:
            condition = cond(transformed_expr_rec)
        except Exception:
            continue
        if condition:
            return op(transformed_expr_rec)
    return transformed_expr_rec


# Cheap subterm closure without checks for sort.
def get_all_subterms(terms):
    """
    Return all subterms of the given set of terms.
    :param terms: set of z3.ExprRef
    :return: set of z3.ExprRef
    """
    subterm_closure = set()
    for term in terms:
        subterm_closure.add(term)
        if term.decl().arity != 0:
            subterm_closure = subterm_closure | get_all_subterms(set(term.children()))
    return subterm_closure
