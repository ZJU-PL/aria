from z3 import *
set_param('model.compact', False)
import re

from aria.quant.fossil.naturalproofs.decl_api import get_recursive_definition, get_vocabulary
# from naturalproofs.uct import get_uct_sort

# from naturalproofs.extensions.finitemodel_utils import transform_fg_universe


###########################
# Support for synthesis modes


class StopProposal(Exception):
    pass


############################
# Support for python models


# Implementing a copy function because dictionary with dictionary entries is not
# copied as expected. The inner dictionaries are stll passed by reference
# Consider doing a more general and systematic deepcopy implementation
def deepcopyModel(model):
    new_model = {}
    for key in model.keys():
        entry = model[key]
        if isinstance(entry,list) or isinstance(entry,dict):
            new_model[key] = entry.copy()
        else:
            new_model[key] = model[key]
    return new_model

##################################
# General unclassified utilities


# Cartesian product of two lists of elements, with a given function applied to
# the pair Default is a + function which will work if defined for the sort of
# list elements
def listProduct(ll1, ll2, combine_func = lambda x,y: x + y):
    return [ combine_func(x,y) for x in ll1 for y in ll2 ]


def getLemmaHeader(lemma, lemma_args):
    lemma_args_str = [ arg.decl().name() for arg in lemma_args ]
    result = re.search('lemma (.*) Bool', lemma)
    params = result.group(1)[1:][:-1]
    params_list = [ i.split(' ')[0] for i in re.findall('\(([^)]+)', params) ]
    header = ''
    for i in range(len(params_list)):
        header += lemma_args_str[i] + ' '
    return '(lemma ' + header[:-1] + ')'


# replace arguments of all instances of any function in replace_fcts
def replaceArgs(lemma, replace_fcts):
    if lemma.children() == [] or replace_fcts == {}:
        return lemma
    elif lemma.decl() in replace_fcts:
        arg0 = replaceArgs(lemma.arg(0), replace_fcts)
        arg1 = replaceArgs(lemma.arg(1), replace_fcts)
        return replace_fcts[lemma.decl()](arg0, arg1)
    else:
        new_args = []
        for i in range(len(lemma.children())):
            new_arg = replaceArgs(lemma.arg(i), replace_fcts)
            new_args += [ new_arg ]
        return lemma.decl()(new_args)


# swap arguments of all instances of any function in swap_fcts
def swapArgs(lemma, swap_fcts):
    if lemma.children() == [] or swap_fcts == {}:
        return lemma
    elif lemma.decl() in swap_fcts:
        arg0 = swapArgs(lemma.arg(0), swap_fcts)
        arg1 = swapArgs(lemma.arg(1), swap_fcts)
        return swap_fcts[lemma.decl()](arg1, arg0)
    else:
        new_args = []
        for i in range(len(lemma.children())):
            new_arg = swapArgs(lemma.arg(i), swap_fcts)
            new_args += [ new_arg ]
        return lemma.decl()(new_args)


# translate output of cvc4 into z3py form
# TODO: abstract this out as general function, not specific to each input
def translateLemma(lemma, lemma_args, addl_decls, swap_fcts, replace_fcts, annctx):
    header = getLemmaHeader(lemma, lemma_args)
    assertion = '(assert ' + header + ')'
    smt_string = lemma + '\n' + assertion
    vocab = get_vocabulary(annctx)
    translate_dict = {v.name() : v for v in vocab}
    translate_dict.update(addl_decls)
    z3py_lemma = parse_smt2_string(smt_string, decls=translate_dict)[0]
    z3py_lemma_replaced = replaceArgs(z3py_lemma, replace_fcts)
    z3py_lemma_fixed = swapArgs(z3py_lemma_replaced, swap_fcts)
    return z3py_lemma_fixed
