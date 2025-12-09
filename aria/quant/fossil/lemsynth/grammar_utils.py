# Some hacks for standardising instantiations/finite model extractions with respect to given grammar
# It is clear that the grammar influences instantiations in a way that has to be uniform in all rounds
# Otherwise soundness bugs crop up

import itertools
import z3

from aria.quant.fossil.naturalproofs.decl_api import get_boolean_recursive_definitions, is_var_decl
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula
from aria.quant.fossil.naturalproofs.prover_utils import instantiate, get_foreground_terms
from aria.quant.fossil.naturalproofs.utils import get_all_subterms


# A set of lemmas whose foreground terms are all the terms that will appear in any proposed lemma
def representative_lemmas(lemma_args, lemma_grammar_terms, annctx, future=False):
    # Construct fake formula with lemma_grammar_terms involving all the terms (assumed to be foreground)
    example_lemma_body = z3.And([t == t for t in lemma_grammar_terms])
    # Construct all possible pfp formulae out of these and collect terms
    # Only boolean valued recursive definitions
    bool_recdefs = get_boolean_recursive_definitions(annctx)
    example_lemmas = set()
    for recdef in bool_recdefs:
        if future:
            # Recursive definitions can have any of the lemma_args as their arguments
            example_bindings = itertools.product(lemma_args, repeat=recdef.arity())
            example_lemmas = example_lemmas.union({z3.Implies(recdef(*binding), example_lemma_body) for binding in example_bindings})
        else:
            # The arguments of the recursive definition will be the first 'arity' variables
            example_lemmas.add(z3.Implies(recdef(*[lemma_args[:recdef.arity()]]), example_lemma_body))
    return example_lemmas


# Terms for instantiation for lemma counterexamples
# Subterm closure of all terms that could appear in any pfp of any lemma
# Given terms in the lemma grammar, this can be computed automatically.
def lemma_instantiation_terms(lemma_args, lemma_grammar_terms, annctx):
    example_lemmas = representative_lemmas(lemma_args, lemma_grammar_terms, annctx=annctx)
    pfp_formulas = {make_pfp_formula(lemma, annctx=annctx) for lemma in example_lemmas}
    instantiation_terms = get_foreground_terms(pfp_formulas, annctx=annctx)
    # get_foreground_terms already performs subterm closure
    return instantiation_terms


# Terms for extracting a finite model from proof attempt of goal
# All terms that could appear in any instantiation of the goal with all possible lemmas
# Applicable only when instantiation mode is depth_one_untracked_lemma_instantiation
def goal_extraction_terms(goal_instantiation_terms, lemma_args, lemma_grammar_terms, annctx):
    example_lemmas = representative_lemmas(lemma_args, lemma_grammar_terms, annctx=annctx)
    # Only the variables in lemma_args can be quantified over
    lemma_params = tuple([lemma_arg for lemma_arg in lemma_args if is_var_decl(lemma_arg)])
    bound_example_lemmas = {(lemma_params, example_lemma) for example_lemma in example_lemmas}
    # Instantiate example lemmas with all possible terms from the instantiation the goal and axioms
    example_instantiations = instantiate(bound_example_lemmas, goal_instantiation_terms)
    example_extraction_terms = get_foreground_terms(example_instantiations, annctx=annctx)
    # Model extraction already performs subterm closure so there is no need to do it here.
    return example_extraction_terms
