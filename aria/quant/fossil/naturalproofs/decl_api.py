# This module defines the API to declare variables and functions in the UCT fragment.

import z3
from aria.quant.fossil.naturalproofs.uct import UCTSort, is_expr_fg_sort, boolsort
from aria.quant.fossil.naturalproofs.AnnotatedContext import (
    AnnotatedContext,
    default_annctx,
)


# Functions to create declarations
def Const(name, uct_sort, annctx=default_annctx):
    """
    Declare a constant with the given name and uct sort.
    :param name: string
    :param uct_sort: naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: z3.ExprRef
    """
    if not isinstance(uct_sort, UCTSort):
        raise TypeError("UCTSort expected.")
    z3const = z3.Const(name, uct_sort.z3sort)
    if not isinstance(annctx, AnnotatedContext):
        raise TypeError("AnnotatedContext expected.")
    # The constant must be tracked as a 0-arity function
    declaration = z3const.decl()
    annctx.add_alias_annotation(declaration, tuple([uct_sort]))
    annctx.add_vocabulary_annotation(declaration)
    return z3const


def Consts(names, uct_sort, annctx=default_annctx):
    """
    Declare a list of constants.
    :param names: string containing all the names separated by a space
    :param uct_sort: naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: list of z3.ExprRef
    """
    if not isinstance(uct_sort, UCTSort):
        raise TypeError("UCTSort expected.")
    z3consts = z3.Consts(names, uct_sort.z3sort)
    if not isinstance(annctx, AnnotatedContext):
        raise TypeError("AnnotatedContext expected.")
    for z3const in z3consts:
        # Constants must be tracked as a 0-arity function
        declaration = z3const.decl()
        annctx.add_alias_annotation(declaration, tuple([uct_sort]))
        annctx.add_vocabulary_annotation(declaration)
    return z3consts


def Var(name, uct_sort, annctx=default_annctx):
    """
    Declare a variable with the given name and uct sort.
    :param name: string
    :param uct_sort: naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: z3.ExprRef
    """
    z3const = Const(name, uct_sort, annctx)
    annctx.add_variable_annotation(z3const)
    return z3const


def Vars(names, uct_sort, annctx=default_annctx):
    """
    Declare a list of variables.
    :param names: string containing all the names separated by a space
    :param uct_sort: naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: list of z3.ExprRef
    """
    z3consts = Consts(names, uct_sort, annctx)
    for z3const in z3consts:
        annctx.add_variable_annotation(z3const)
    return z3consts


def Function(name, *uct_signature, annctx=default_annctx):
    """
    Declare an uninterpreted function symbol. The signature is given as input-sort, input-sort...output-sort
    :param name: string
    :param uct_signature: tuple of naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: z3.FuncDeclRef
    """
    if not all([isinstance(sig, UCTSort) for sig in uct_signature]):
        raise TypeError("UCTSort expected.")
    if not isinstance(annctx, AnnotatedContext):
        raise TypeError("AnnotatedContext expected.")
    if len(uct_signature) < 2:
        raise ValueError(
            "There must be atleast one input sort and exactly one output sort."
        )
    z3sig = [sig.z3sort for sig in uct_signature]
    z3func = z3.Function(name, *z3sig)
    annctx.add_alias_annotation(z3func, uct_signature)
    annctx.add_vocabulary_annotation(z3func)
    return z3func


def RecFunction(name, *uct_signature, annctx=default_annctx):
    """
    Declare a recursively defined function symbol.
    :param name: string
    :param uct_signature: tuple of naturalproofs.uct.UCTSort
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: z3.FuncDeclRef
    """
    # Currently defaults to calling Function as recursive functions are not tracked in a separate way.
    return Function(name, *uct_signature, annctx=annctx)


def AddRecDefinition(recdef, formal_params, body, annctx=default_annctx):
    """
    Add a definition to a recursive function symbol. The function symbol must be declared and tracked before a definiiton
    can be added.
    The definition is given in terms of a tuple of formal parameters that are themselves declared constants, and the body
    is a z3.ExprRef object constructed from these constants and other declared/built-in functions.
    :param recdef: z3.FuncDeclRef
    :param formal_params: tuple of z3.ExprRef (currently only z3.ArithRef)
    :param body: z3.ExprRef
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: None
    """
    if not isinstance(formal_params, tuple) and isinstance(formal_params, z3.ExprRef):
        # Only one formal parameter
        formal_params = (formal_params,)
    if not all(is_var_decl(v, annctx) for v in formal_params):
        raise ValueError(
            "All formal parameters must be variables declared using naturalproofs.decl_api.{Var, Vars}."
        )
    if not annctx.is_tracked_vocabulary(recdef):
        raise ValueError(
            "Function symbol must be declared using naturalproofs.decl_api.Function"
        )
    if len(formal_params) != recdef.arity():
        raise ValueError(
            "Number of formal parameters does not match arity of function symbol."
        )
    # Check that all formal parameters are of the foreground sort.
    # Arguments of other sorts are not supported.
    elif not all([is_expr_fg_sort(param, annctx) for param in formal_params]):
        raise TypeError("All formal parameters can only be of the foreground sort.")
    if not isinstance(body, z3.ExprRef):
        raise TypeError("ExprRef expected.")
    # TODO: check that the definition is of the supported form: positive recursive mentions, for example.
    annctx.add_recdef_annotation((recdef, formal_params, body))


def AddAxiom(formal_params, body, annctx=default_annctx):
    """
    Add an axiom with respect to which the reasoning must be performed.
    The axiom is given in terms of a tuple of formal parameters that are themselves declared constants, and the body
    is a z3.ExprRef object constructed from these constants and other declared/built-in functions. If the axiom does not
    take any parameters, the first argument is ().
    :param formal_params: tuple of z3.ExprRef (currently only z3.ArithRef)
    :param body: z3.ExprRef
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: None
    """
    if not isinstance(formal_params, tuple) and isinstance(formal_params, z3.ExprRef):
        # Only one formal parameter
        formal_params = (formal_params,)
    if not all(is_var_decl(v, annctx) for v in formal_params):
        raise ValueError(
            "All formal parameters must be variables declared using naturalproofs.decl_api.{Var, Vars}."
        )
    # Check that all formal parameters are of the foreground sort.
    # Arguments of other sorts are not supported.
    if not all([is_expr_fg_sort(param, annctx) for param in formal_params]):
        raise TypeError("All formal parameters can only be of the foreground sort.")
    if not isinstance(body, z3.ExprRef):
        raise TypeError("ExprRef expected.")
    annctx.add_axiom_annotation((formal_params, body))


# Utility functions to manipulate declarations
def get_vocabulary(annctx=default_annctx):
    """
    Returns the set of all the declarations tracked by annctx
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: set of z3.FuncDeclRef
    """
    return annctx.get_vocabulary_annotation()


def get_uct_signature(funcdeclref, annctx=default_annctx):
    """
    Returns the uct signature of the given function if tracked by annctx.
    :param funcdeclref: z3.FuncDeclRef
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: tuple of naturalproofs.uct.UCTSort or None
    """
    return annctx.read_alias_annotation(funcdeclref)


def get_decl_from_name(declname, annctx=default_annctx):
    """
    Returns the declaration whose name is declname if it is tracked by annctx.
    :param declname: string
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: z3.FuncDeclRef or None
    """
    vocabulary = get_vocabulary(annctx)
    return next((decl for decl in vocabulary if decl.name() == declname), None)


def is_var_decl(exprref, annctx=default_annctx):
    """
    Determines if the given expression is a variable tracked by annctx, or a constant tracked by annctx in the alias
    annotation. Returns None if neither is true.
    :param exprref: z3.ExprRef
    :param annctx: naturalproofs.AnnnotatedContext.AnnotatedContext
    :return: bool or None
    """
    if not isinstance(exprref, z3.ExprRef):
        return None
    if exprref.decl().arity() != 0:
        return None
    if not annctx.is_tracked_alias(exprref.decl()):
        return None
    return exprref in annctx.get_variable_annotation()


def get_recursive_definition(recdef, alldefs=False, annctx=default_annctx):
    """
    Looks up the definition of the function symbol from the set of recursive definitions in the annctx context.
    Returns None if no definition exists in the context.
    If the alldefs is true, then all recursive definitions are returned.
    :param recdef: z3.FuncDeclRef
    :param alldefs: bool
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: (recdef, tuple of z3.ExprRef, z3.ExprRef), or a set of such triples, or None
    """
    recdef_set = annctx.get_recdef_annotation()
    if alldefs:
        return recdef_set
    else:
        if not annctx.is_tracked_vocabulary(recdef):
            raise ValueError(
                "Function symbol must be declared using naturalproofs.decl_api.Function"
            )
        return next(
            (definition for definition in recdef_set if recdef == definition[0]), None
        )


def get_boolean_recursive_definitions(annctx=default_annctx):
    """
    Returns sorted list of all recursive definitions that return a boolean.
    Returns [] if no recursive definitions exist in the context.
    Recursive definitions are sorted by .name() attribute
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: list of z3.FuncDeclRef
    """
    recdef_set = annctx.get_recdef_annotation()
    recs = set(x[0] for x in recdef_set)
    sorted_recs = sorted(recs, key=lambda x: x.name())
    bool_recs = [x for x in sorted_recs if get_uct_signature(x)[-1] == boolsort]
    return bool_recs


def get_all_axioms(annctx=default_annctx):
    """
    Returns all axioms tracked by annctx.
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :return: (tuple of z3.ExprRef, z3.ExprRef)
    """
    return annctx.get_axiom_annotation()
