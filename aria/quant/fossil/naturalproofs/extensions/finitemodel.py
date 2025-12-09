"""
This module defines the class FiniteModel representing finite models extracted from smt models witnessing a proof
failure using the natural proofs package.

Explanation of object attributes:
- Object attributes
-- finitemodel: the finite model extracted from an smt model.
-- smtmodel: the smt model that was used for extraction.
-- vocabulary: the vocabulary of constants and functions represented in the finite model.
-- annctx: the annotated context in which the foreground sort annotation is tracked.
-- offset: the 'offset' added to the foreground universe after extraction. Offset is initially 0. If at any point the
offset is not 0, then the value of any foreground term present in the finite model is its value as given by the smt
 model, plus the offset.
--- fg_universe: the set of all foreground elements present in the finite model.
- Logging attributes
-- extraction_terms: the terms used for finite model extraction at the time of creation.
- Caching attributes
-- recompute_offset: whether the model is already offset from the true values in the smt model, or if future offset
computations must add to the current offset. Used for 'caching' offset models without needing further offsets unless
explicitly specified. Set to 'True' by default, so all offset computations will have an effect.
"""

import itertools
import copy
import z3
# model.compact should be turned off to not get lambdas, only actual arrays/sets.
z3.set_param('model.compact', False)

from aria.quant.fossil.naturalproofs.AnnotatedContext import default_annctx
from aria.quant.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort
from aria.quant.fossil.naturalproofs.decl_api import get_vocabulary, get_uct_signature
from aria.quant.fossil.naturalproofs.prover_utils import get_foreground_terms
from aria.quant.fossil.naturalproofs.extensions.finitemodel_utils import transform_fg_universe, collect_fg_universe


class FiniteModel:
    def __init__(self, smtmodel, terms, vocabulary=None, annctx=default_annctx):
        """
        Finite model creation.
        Foreground universe of extracted model corresponds to terms with subterm-closure. If vocabulary is not
        specified, the entire vocabulary tracked in annctx is used.
        :param smtmodel: z3.ModelRef
        :param terms: set of z3.ExprRef
        :param vocabulary: set of z3.ExprRef
        :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext

        This function also defines the format of finite models.
        Given a vocabulary of functions f1, f2, ..fm, of arity a1, a2, ...am, the model is as follows:
        model :: dict {'fk' : dict_fk}
        where 'fk' is some representation of the function fk, and
        dict_fk :: dict {(e1, e2,... ek) : fk(e1, e2, ...ek)}
        where (e1, e2, ...ek) is a tuple such that e1, e2,.. etc are concrete values in python that are
        dependent on the domain and range sorts of fk.
        In particular if the arity k is 0, then dict_fk will be of the following form:
        dict_fk :: dict {() : fk()}

        These models are meant to be partial models, and in general it will not be possible to evaluate an arbitrary
        formula on such a model, just those that are quantifier-free with foreground terms in the set of terms used
        to extract the finite model.
        """
        model = dict()
        # TODO: VERY IMPORTANT: hack to include those terms that are directly given as integers or integer expressions
        elems = {smtmodel.eval(term, model_completion=True) for term in terms}
        # TODO: the assumption is that uninterpreted functions have arguments only from the foreground sort. Must handle
        #  cases where uninterpreted functions have arguments in other domains, primarily integers.
        # Subterm-close the given terms assuming one-way functions
        # get_foreground_terms already performs subterm closure
        subterm_closure = get_foreground_terms(terms, annctx=annctx)
        elems = elems | {smtmodel.eval(term, model_completion=True) for term in subterm_closure}
        if vocabulary is None:
            vocabulary = get_vocabulary(annctx)
        for func in vocabulary:
            arity = func.arity()
            *input_signature, output_sort = get_uct_signature(func, annctx)
            # Only supporting uninterpreted functions with input arguments all from the foreground sort
            if not all(sig == fgsort for sig in input_signature):
                raise ValueError('Function with input(s) not from the foreground sort. Unsupported.')
            func_key_repr = model_key_repr(func)
            # Distinguish common cases for faster execution
            if arity == 0:
                model[func_key_repr] = {(): _extract_value(smtmodel.eval(func(), model_completion=True), output_sort)}
            elif arity == 1:
                model[func_key_repr] = {
                    (_extract_value(elem, fgsort),): _extract_value(smtmodel.eval(func(elem), model_completion=True),
                                                                    output_sort) for elem in elems}
            else:
                func_dict = dict()
                args = itertools.product(elems, repeat=arity)
                for arg in args:
                    arg_value = tuple(_extract_value(component, fgsort) for component in arg)
                    func_dict[arg_value] = _extract_value(smtmodel.eval(func(*arg), model_completion=True), output_sort)
                model[func_key_repr] = func_dict

        # Object attributes
        self.finitemodel = model
        self.smtmodel = smtmodel
        self.vocabulary = vocabulary
        self.annctx = annctx
        self.offset = 0
        self.fg_universe = collect_fg_universe(self.finitemodel, self.annctx)
        # Logging attributes
        self.extraction_terms = subterm_closure
        # Caching attributes
        self.recompute_offset = True

    def copy(self):
        """
        Custom copy implementation that 'deepcopies' the finitemodel attribute alone.
        :return: naturalproofs.extensions.finitemodel.FiniteModel
        """
        finitemodel_copy = copy.deepcopy(self.finitemodel)
        # Blank object
        # Init as defined will not do anything but assign the smtmodel field
        copy_object = FiniteModel(smtmodel=self.smtmodel, terms={}, vocabulary=[], annctx=None)
        copy_object.finitemodel = finitemodel_copy
        copy_object.smtmodel = self.smtmodel
        copy_object.vocabulary = self.vocabulary
        copy_object.annctx = self.annctx
        copy_object.offset = self.offset
        copy_object.fg_universe = self.fg_universe
        copy_object.extraction_terms = self.extraction_terms
        copy_object.recompute_offset = self.recompute_offset
        return copy_object

    # Some common functions on finite models
    def get_fg_elements(self):
        # fg_elem_set = set()
        # _ = transform_fg_universe(finite_model, lambda x: (fg_elem_set.add(x), x)[1], annctx)
        # return fg_elem_set
        return self.fg_universe

    def add_fg_element_offset(self, offset_value):
        if self.recompute_offset:
            self.finitemodel = transform_fg_universe(self.finitemodel, lambda x: x + offset_value, self.annctx)
            self.offset = self.offset + offset_value
            self.fg_universe = {elem + offset_value for elem in self.fg_universe}


# Helper functions for extract_finite_model
# Representation of keys in the finite model top-level dictionary.
def model_key_repr(funcdeclref):
    # Should be equivalent to naturalproofs.AnnotatedContext._alias_annotation_key_repr(funcdeclref)
    # For z3.FuncDeclRef objects, this is almost always equal to name()
    return funcdeclref.name()


def _extract_value(value, uct_sort):
    """
    Converts the value of a concrete constant represented as a z3.ExprRef into a simple python type
    that can be explicitly manipulated.
    The explicit conversion scheme is as follows:
    fgsort -> int
    fgsetsort -> set of int
    intsort -> int
    intsetsort -> set of int
    boolsort -> bool
    :param value: z3.ExprRef
    :param uct_sort: uct.UCTSort
    :return: any
    """
    # value assumed to be such that value.decl().arity == 0
    if uct_sort == boolsort:
        # value is either BoolVal(True) or BoolVal(False)
        return z3.is_true(value)
    elif uct_sort == fgsort or uct_sort == intsort:
        # value is an IntVal(v) for some number v
        return value.as_long()
    elif uct_sort == fgsetsort or uct_sort == intsetsort:
        # value is a set of integers.
        # model.compact has been disabled (see top of file). ArrayRef should not have lambdas in it.
        if not isinstance(value, z3.ArrayRef):
            raise ValueError('Something is wrong. Model is returning lambdas instead of arrays.')
        # iteratively deconstruct the expression to build the set of python numbers.
        extracted_set = set()
        value = value.__deepcopy__()
        while True:
            if z3.is_K(value):
                # Base case. Either full set or empty set.
                if z3.simplify(value[0]):
                    # Full set of integers. Raise exception.
                    raise ValueError('Model assigned infinite sets to some interpretations. Unsupported.')
                else:
                    return extracted_set
            elif z3.is_store(value):
                remaining_set, entry, if_belongs = value.children()
                value = remaining_set
                extracted_set = extracted_set | ({entry.as_long()} if z3.is_true(if_belongs) else {})
            else:
                raise ValueError('ArrayRef is constructed with neither Store nor K. Possible multidimensional arrays. '
                                 'Unsupported.')
    else:
        raise ValueError('UCT Sort type not supported for extraction of models.')


def recover_value(value, uct_sort):
    """
    Inverse of _extract_value. Given a pythonic value, returns a z3 embedding of it depending on its uct sort. The
    explicit embedding scheme is as follows:
    fgsort -> z3.ArithRef
    fgsetsort -> z3.ArrayRef
    intsort -> z3.ArithRef
    intsetsort -> z3.ArrayRef
    boolsort -> z3.BoolRef
    :param value: any
    :param uct_sort: naturalproofs.uct.UCTSort
    :return: z3.ExprRef
    """
    # TODO: typecheck all the arguments
    if uct_sort in {fgsort, intsort}:
        return z3.IntVal(value)
    elif uct_sort == boolsort:
        return z3.BoolVal(value)
    elif uct_sort in {fgsetsort, intsetsort}:
        expr = z3.EmptySet(z3.IntSort())
        for elem in value:
            expr = z3.SetAdd(expr, z3.IntVal(elem))
    else:
        raise ValueError('Sort not supported. Check for a list of available sorts in the naturalproofs.uct module.')
