"""
Some utilities for manipulating finite models as defined in naturalproofs.extensions.finitemodel.
"""

import z3

from aria.quant.fossil.naturalproofs.uct import fgsort, fgsetsort
from aria.quant.fossil.naturalproofs.decl_api import get_decl_from_name, get_uct_signature


def collect_fg_universe(finite_model, annctx):
    """
    Returns all elements of the foreground universe present in finite_model. All vocabulary entries must be tracked by
    annctx.
    :param finite_model: dict {string -> dict {tuple of any -> any}}
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: set of any
    """
    fg_elem_set = set()
    vocab_keys = finite_model.keys()
    for vocab_key in vocab_keys:
        vocab_decl = get_decl_from_name(vocab_key, annctx)
        arity = vocab_decl.arity()
        uct_signature = get_uct_signature(vocab_decl, annctx)
        if all(sig not in {fgsort, fgsetsort} for sig in uct_signature):
            # No foreground sort elements anywhere. No collection required
            continue
        args = finite_model[vocab_key].keys()
        for arg in args:
            for i in range(arity):
                fg_elem_set = fg_elem_set | _collect_value(arg[i], uct_signature[i])
                fg_elem_set = fg_elem_set | _collect_value(finite_model[vocab_key][arg], uct_signature[-1])
    return fg_elem_set


def transform_fg_universe(finite_model, lambdafunc, annctx):
    """
    Applies the given function lambdafunc to all elements of the foreground universe present in finite_model. All
    vocabulary entries must be tracked by annctx.
    :param finite_model: dict {string -> dict {tuple of any -> any}}
    :param lambdafunc: function
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: dict {string -> dict {tuple of any -> any}}
    """
    vocab_keys = finite_model.keys()
    for vocab_key in vocab_keys:
        vocab_decl = get_decl_from_name(vocab_key, annctx)
        arity = vocab_decl.arity()
        uct_signature = get_uct_signature(vocab_decl, annctx)
        if all(sig not in {fgsort, fgsetsort} for sig in uct_signature):
            # No foreground sort elements anywhere. No transformation required
            continue
        new_vocab_key_dict = dict()
        args = finite_model[vocab_key].keys()
        for arg in args:
            new_arg = tuple(_transform_value(arg[i], uct_signature[i], lambdafunc) for i in range(arity))
            new_out = _transform_value(finite_model[vocab_key][arg], uct_signature[-1], lambdafunc)
            new_vocab_key_dict[new_arg] = new_out
        finite_model[vocab_key] = new_vocab_key_dict
    return finite_model


# def evaluate(finite_model, expr, symbolic_compute):
#     """
#     Evaluates the given expression on a finite model as defined in naturalproofs.extensions.finitemodel. An exception
#     is raised if any sub-expression cannot be evaluated due to partial interpretations, unless symbolic_compute is set
#     to True.
#     Returns a pythonic value type if symbolic_compute is False, otherwise returns z3.ExprRef.
#     :param finite_model: dict {string -> dict {tuple of any -> any}}
#     :param expr: z3.ExpRef
#     :param symbolic_compute: bool
#     :return: any
#     """
#     return None


# Auxiliary functions
# Helper function for collect_fg_universe
def _collect_value(value, uctsort):
    if uctsort == fgsort:
        return {value}
    elif uctsort == fgsetsort:
        return value
    else:
        return set()


# Helper function for transform_fg_universe
def _transform_value(value, uctsort, lambdafunc):
    if uctsort == fgsort:
        return lambdafunc(value)
    elif uctsort == fgsetsort:
        return set(lambdafunc(x) for x in value)
    else:
        return value


# Helper function for evaluate
# def _evaluate_interpreted_function(funcdeclref):
#
