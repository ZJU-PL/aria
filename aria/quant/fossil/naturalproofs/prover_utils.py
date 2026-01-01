# This module defines some utilities needed to create and run a natural proofs solver.
# The main natural proofs module itself will contain various instantiation strategies, configuration options, etc.

import z3
import itertools

from aria.quant.fossil.naturalproofs.AnnotatedContext import default_annctx
from aria.quant.fossil.naturalproofs.uct import is_expr_fg_sort
from aria.quant.fossil.naturalproofs.decl_api import get_recursive_definition
from aria.quant.fossil.naturalproofs.utils import apply_bound_formula


def instantiate(bound_formulas, terms):
    """
    Instantiates every formula in bound_formulas with given terms. Each bound formula is a pair
    (tuple of formal parameters, formula in terms of parameters).
    :param bound_formulas: (tuple of z3.ExprRef, z3.ExprRef) or a set of such pairs
    :param terms: set of z3.ExprRef or tuple of z3.ExprRef
    :return: set of z3.ExprRef
    """
    if isinstance(bound_formulas, tuple):
        # Only one bound formula given
        bound_formulas = {bound_formulas}
    instantiated_set = set()
    terms = list(terms)
    for bound_formula in bound_formulas:
        formal_params, body = bound_formula
        arity = len(formal_params)
        # The arguments are all possible tuples of terms whose length is arity
        # TODO: make this step more efficient by sorting bound_formulas by arity or using numpy.product
        if isinstance(terms[0], z3.ExprRef):
            # Only individual terms are given. Make argument tuples with them.
            arg_tuples = itertools.product(terms, repeat=arity)
        else:
            # Check that all the tuples are of the same arity
            if not all(len(term_tup) == arity for term_tup in terms):
                raise ValueError(
                    "Arguments must be the same length as the arity of the bound formula to be applied."
                )
            arg_tuples = terms
        for arg_tuple in arg_tuples:
            instantiated_set.add(apply_bound_formula(bound_formula, arg_tuple))
    return instantiated_set


def _get_foreground_terms_aux(expr, annctx):
    # Auxiliary function for get_foreground_terms collecting foreground terms from one expression.
    # Recursively break down expression and check if it is of the foreground sort. If it is, add it to the accumulator.
    fg_set = {expr} if is_expr_fg_sort(expr, annctx) else set()
    arity = expr.decl().arity()
    if arity == 0:
        return fg_set
    else:
        children = expr.children()
        for child in children:
            fg_set = fg_set.union(_get_foreground_terms_aux(child, annctx))
    return fg_set


def get_foreground_terms(exprs, annctx):
    """
    Return the subterm-closed set of terms in any expression in exprs that are of the foreground sort.
    :param exprs: z3.ExprRef or set of z3.ExprRef
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: set of z3.ExprRef
    """
    if isinstance(exprs, z3.ExprRef):
        # Only one expression given
        fg_set = _get_foreground_terms_aux(exprs, annctx)
    else:
        fg_set = set()
        for expr in exprs:
            if not isinstance(expr, z3.ExprRef):
                raise TypeError("ExprRef expected")
            fg_set = fg_set.union(_get_foreground_terms_aux(expr, annctx))
    return fg_set


def _make_recdef_unfoldings_aux(recdef_triple):
    # Auxiliary function for make_recdef_unfoldings making an unfolding from one recursive definition.
    # Simply unpack the triple, construct the unfolding, and repack it into a pair.
    recdef, formal_params, body = recdef_triple
    if not isinstance(recdef, z3.FuncDeclRef):
        raise TypeError("FuncDeclRef expected.")
    # The unfolding is simply the fact that the recursive function is equal to its body on the formal parameters.
    unfolding = {recdef: (formal_params, recdef(*formal_params) == body)}
    return unfolding


def make_recdef_unfoldings(recursive_definitions):
    """
    Make a bound formula from the given recursive definition that corresponds to its 'unfolding' once on given input.
    :param recursive_definitions: (z3.FuncDeclRef, tuple of z3.ExprRef, z3.ExprRef) or a set of such triples
    :return: dict {z3.FuncDeclRef : (tuple of z3.ExprRef, z3.ExprRef)}
    """
    if not isinstance(recursive_definitions, set):
        # Only one recursive definition given
        return _make_recdef_unfoldings_aux(recursive_definitions)
    else:
        unfoldings_dict = dict()
        for recursive_definition in recursive_definitions:
            unfoldings_dict = {
                **unfoldings_dict,
                **_make_recdef_unfoldings_aux(recursive_definition),
            }
        return unfoldings_dict


def _get_recdef_applications_aux(expr, recdefs):
    decl = expr.decl()
    arity = decl.arity()
    applications = []
    if arity > 0:
        applications = []
        if decl in recdefs:
            applications = applications + [(decl, tuple(expr.children()))]
        for child in expr.children():
            applications = applications + _get_recdef_applications_aux(child, recdefs)
    return applications


def get_recdef_applications(exprs, annctx):
    """
    Return a dictionary containing tuples that occur under applications of recursive definitions. The dictionary
    is indexed by recursive definitions.
    :param exprs: z3.ExprRef or set of z3.ExprRef
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: dict {z3.FuncDeclRef : list of tuples of z3.ExprRef}
    """
    recdef_triples = get_recursive_definition(None, alldefs=True, annctx=annctx)
    recdefs = [triple[0] for triple in recdef_triples]
    if isinstance(exprs, z3.ExprRef):
        # Only one expression given
        result = _get_recdef_applications_aux(exprs, recdefs)
    else:
        result = []
        for expr in exprs:
            if not isinstance(expr, z3.ExprRef):
                raise TypeError("ExprRef expected")
            result = result + _get_recdef_applications_aux(expr, recdefs)
    # Make the list of applications into a dictionary
    applications = dict()
    for recdef, application in result:
        if recdef not in applications:
            applications[recdef] = []
        applications[recdef].append(application)
    return applications
