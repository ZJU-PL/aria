import subprocess
import itertools
import time
import warnings

from z3 import *
set_param('model.compact', False)

import aria.quant.fossil.lemsynth.options as options
from aria.quant.fossil.lemsynth.induction_constraints import generate_pfp_constraint
from aria.quant.fossil.lemsynth.cvc4_compliance import cvc4_compliant_formula_sexpr
from aria.quant.fossil.lemsynth.ProcessStreamer import ProcessStreamer, Timeout
from aria.quant.fossil.lemsynth.utils import StopProposal

from aria.quant.fossil.naturalproofs.decl_api import get_uct_signature, get_boolean_recursive_definitions, is_expr_fg_sort
from aria.quant.fossil.naturalproofs.prover import NPSolver
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions
from aria.quant.fossil.naturalproofs.extensions.finitemodel import recover_value
from aria.quant.fossil.naturalproofs.extensions.finitemodel import FiniteModel
from aria.quant.fossil.naturalproofs.decl_api import get_vocabulary, is_var_decl


# Add constraints from each model into the given solver
# Look through model's function entries and adds each input-output constraint
def modelToSolver(model, vocab, sol, annctx):
    for fct in vocab:
        arity = fct.arity()
        if arity == 0:
            # TODO: handle constant symbols
            # constant symbol
            continue
        else:
            fct_name = fct.name()
            uct_signature = get_uct_signature(fct, annctx)
            for input_arg in model[fct_name].keys():
                output_value = model[fct_name][input_arg]
                if isinstance(output_value, set):
                    output_value_converted = recover_value(output_value, uct_signature[-1])
                else:
                    output_value_converted = output_value

                if isinstance(input_arg, tuple):
                    # arg must be unpacked as *arg before constructing the Z3 term
                    sol.add(fct(*input_arg) == output_value_converted)
                else:
                    sol.add(fct(input_arg) == output_value_converted)

def translateSet(s, fct_range):
    out = ''
    for i in s:
        if i < 0:
            val = '(- ' + str(i * -1) + ')'
        else:
            val = str(i)
        out += '(insert ' + val + ' '
    if options.synthesis_solver == options.minisy:
        out += 'empIntSet'
    else:
        out += '(as emptyset (' + fct_range + '))'
    for i in s:
        out += ')'
    return out


# translate tuple of args to conjunction of equalities in smt format
def translateArgs(elt):
    out = ''
    for i in range(len(elt)):
        if elt[i] < 0:
            val = '(- ' + str(elt[i] * -1) + ')'
        else:
            val = str(elt[i])
        out += '(= x!' + str(i) + ' ' + val + ') '
    return out[:-1]


# get header of set function
def getHeader(fct, fct_range):
    out = '(define-fun ' + fct.name() + ' ('
    for i in range(0, fct.arity()):
        out += '(x!' + str(i) + ' ' + str(fct.domain(i)) + ') '
    out = out[:-1] + ') '
    out += '(' + fct_range + ')'
    return out


# translate models of fully evaluated sets to smtlib format
def translateModelsSets(models, set_defs):
    out = ''
    for fct in set_defs:
        fct_name = fct.name()
        fct_range = 'Set ' + str(fct.range().domain())
        curr_fct = getHeader(fct, fct_range) + '\n'
        body = ''
        for model in models:
            curr_model_body = ''
            for elt in model[fct_name]:
                args = translateArgs(elt)
                set_translate = translateSet(model[fct_name][elt], fct_range)
                curr_model_body += '  (ite (and ' + args + ') ' + set_translate + '\n'
            body += curr_model_body
        if options.synthesis_solver == options.minisy:
            body += '  empIntSet'
        else:
            body += '  (as emptyset (' + fct_range + '))'
        for model in models:
            for elt in model[fct_name]:
                body += ')'
        curr_fct += body + ')\n\n'
        out += curr_fct
    return out


# Generate single model from a given list of models
# Returns the definitions for functions and recdefs.
# TODO: consider not using z3 for this and just generating the definitions using python code
def sygusBigModelEncoding(models, vocab, set_defs, annctx):
    sol = Solver()
    for model in models:
        modelToSolver(model, vocab, sol, annctx)
    sol.check()
    m = sol.model()
    set_encodings = translateModelsSets(models, set_defs)
    return set_encodings + m.sexpr()


# Generate constraints corresponding to false model for SyGuS
def generateConstraints(model, lemma_args, terms, is_true_constraint, annctx, instantiations=None):
    const = [arg for arg in lemma_args if not is_var_decl(arg, annctx)]
    const_values = [model.smtmodel.eval(cs, model_completion=True).as_long() + (model.offset if is_expr_fg_sort(cs, annctx) else 0) for cs in const]
    const_values = ['(- ' + str(cv * -1) + ')' if cv < 0 else str(cv) for cv in const_values]
    const_values = ' '.join(const_values)
    constraints = ''
    lemma_arity = len(lemma_args) - len(const)
    eval_terms = {model.smtmodel.eval(term, model_completion=True).as_long() + model.offset for term in terms}

    if instantiations is None:
        instantiations = itertools.product(eval_terms, repeat=lemma_arity)
    for arg in instantiations:
        curr = ''
        recs = get_boolean_recursive_definitions()
        arg_str = [str(elt) for elt in arg]
        for i in range(len(recs)):
            rec_arity = recs[i].arity()
            rswitch = '(= rswitch {})'.format(i)
            # Assuming first 'arity' arguments of lemma variables are arguments for recursive definition
            lhs = '({} {})'.format(recs[i].name(), ' '.join(arg_str[:rec_arity]))
            rhs = '(lemma {} {})'.format(' '.join(arg_str), const_values)
            if is_true_constraint:
                curr_constraint = '(=> {} (=> {} {}))\n'.format(rswitch, lhs, rhs)
            else:
                curr_constraint = '(=> {} (not (=> {} {})))\n'.format(rswitch, lhs, rhs)
            curr = curr + curr_constraint
        constraints = constraints + '(and {})\n'.format(curr)
    if is_true_constraint:
        out = '(constraint (and {}))'.format(constraints)
    else:
        out = '(constraint (or {}))'.format(constraints)
    return out


def generateCexConstraints(model, lemma_args, annctx):
    constraints = ''
    recs = get_boolean_recursive_definitions()
    # TODO: NOTE: only one universally quantified variable in desired lemma for now
    for i in range(len(recs)):
        pfp_formula = generate_pfp_constraint(recs[i], lemma_args, model, annctx)
        pfp_formula_sexpr = cvc4_compliant_formula_sexpr(pfp_formula)
        curr_constraint = '(=> (= rswitch {0}) {1})'.format(i, pfp_formula_sexpr)
        constraints = constraints + curr_constraint
    out = '(constraint (and {0}))\n'.format(constraints)
    return out


# Generate constraints corresponding to counterexample models
def generateAllCexConstraints(models, lemma_args, annctx):
    out = ''
    for model in models:
        out = out + generateCexConstraints(model, lemma_args, annctx)
    return out


# preamble for running with z3 (using arrays instead of sets)
def z3Preamble():
    insert_def = '(define-fun insert ((x Int) (y (Array Int Bool))) (Array Int Bool)\n'
    insert_def += '(store y x true)\n)'
    member_def = '(define-fun member ((x Int) (y (Array Int Bool))) Bool\n'
    member_def += '(select y x)\n)'
    empset_def = '(define-fun empIntSet () (Array Int Bool)\n'
    empset_def += '((as const (Array Int Bool)) false)\n)'
    return insert_def + '\n' + member_def + '\n' + empset_def + '\n'


# Handler for converting synthesis query outputs to a generator that yields lemmas
# TODO: Move cvc4 compliance forth-and-back and lemma translation into this module
def process_synth_results(iterable):
    try:
        # Each result comes in pairs of the lemma rhs (body) and the lemma lhs (called rswitch)
        # Looping through iterable/generator two entries at a time: standard grouper recipe from itertools docs
        yield from itertools.zip_longest(*([iterable]*2), fillvalue=None)
    # Special exception class to handle graceful termination of synthesis process(es)
    except StopProposal:
        # Cannot ascertain the type of the iterable to be a generator in a clean way
        # Check minimally for handling lists and throw the exception up to the iterable otherwise
        if type(iterable) != list:
            try:
                iterable.throw(StopProposal)
            except StopIteration:
                pass
    except Timeout:
        # If the generators raises a timeout then there are no more lemmas
        pass
    return


# write output to a file that can be parsed by CVC4 SyGuS
def getSygusOutput(lemmas, lemma_args, goal, problem_instance_name, grammar_string, config_params, annctx):
    # Make log folder if it does not exist already
    os.makedirs(options.log_file_path, exist_ok=True)

    out_file = '{}/out_{}.sy'.format(options.log_file_path, problem_instance_name)

    goal_fo_solver = config_params.get('goal_solver', None)
    if goal_fo_solver is None:
        raise Exception('Something is wrong. A fixed solver object for the goal is needed. Consult an expert.')
    goal_npsolution = goal_fo_solver.solve(goal, lemmas)
    if not goal_npsolution.if_sat:
        # Lemmas generated up to this point are useful. Wrap up processes and exit.
        print('Goal has been proven. Lemmas used to prove goal:')
        for lemma in lemmas:
            print(lemma[1])
        if options.analytics:
            if options.verbose >= 0:
                print('Total lemmas proposed: ' + str(config_params['analytics']['total_lemmas']))
            if options.streaming_synthesis_swtich:
                total_time = config_params['analytics']['time_charged'] + config_params['analytics']['lemma_time']
                if options.verbose >= 0:
                    print('Total time charged: ' + str(total_time) + 's')
        exit(0)

    # Temprarily disabling caching or overriding of goal instantiation or extraction terms
    # goal_extraction_terms = config_params.get('goal_extraction_terms', None)
    # if goal_extraction_terms is not None:
    #     if options.debug:
    #         # Goal extraction terms must be a superset of actual extraction terms
    #         # Otherwise finite model extraction will not work
    #         remaining_terms = goal_npsolution.extraction_terms - goal_extraction_terms
    #         if remaining_terms != set():
    #             raise ValueError('Lemma terms is too small. '
    #                              'Terms remaining after instantiation: {}'.format(remaining_terms))
    #     else:
    #         warnings.warn('The set of terms in the proof of the goal is likely to vary. '
    #                       'Tool may produce false negatives.')
    #         goal_extraction_terms = goal_npsolution.extraction_terms
    # goal_instantiation_terms = config_params.get('goal_npsolution_instantiation_terms',
    #                                              goal_npsolution.instantiation_terms)

    goal_instantiation_terms = goal_npsolution.instantiation_terms
    goal_extraction_terms = goal_npsolution.extraction_terms
    false_finitemodel = FiniteModel(goal_npsolution.model, goal_extraction_terms, annctx=annctx)

    # Adding offsets to make sure: (i) all elements in all models are positive (ii) models do not overlap
    # Making the universe of the false model positive
    false_model_fg_universe = false_finitemodel.get_fg_elements()
    non_negative_offset = min(false_model_fg_universe)
    if non_negative_offset >= 0:
        non_negative_offset = 0
    else:
        false_finitemodel.recompute_offset = True
        false_finitemodel.add_fg_element_offset(abs(non_negative_offset))
    false_model_relative_offset = max(false_model_fg_universe) + abs(non_negative_offset) + 1

    # Extract counterexample models from config_params with default value being []
    cex_models = config_params.get('cex_models', [])

    # Add counterexample models to all models if there are any
    accumulated_offset = false_model_relative_offset
    cex_models_with_offset = []
    for cex_model in cex_models:
        # Deepcopy the countermodels so the originals are not affected
        cex_offset_model = cex_model.copy()
        # Make the universe of the model positive and shift the model by accumulated offset
        cex_model_universe = cex_offset_model.get_fg_elements()
        non_negative_offset = min(cex_model_universe)
        if non_negative_offset >= 0:
            non_negative_offset = 0
        cex_offset_model.add_fg_element_offset(abs(non_negative_offset) + accumulated_offset)
        # Compute new accumulated offset
        accumulated_offset = max(cex_model_universe) + abs(non_negative_offset) + accumulated_offset + 1
        # Add model to cex_models_with_offset
        cex_models_with_offset = cex_models_with_offset + [cex_offset_model]
    cex_models = cex_models_with_offset

    # Add true counterexample model to true models if use_cex_true_models is True
    true_cex_models = config_params.get('true_cex_models', [])
    if true_cex_models:
        true_cex_models_with_offset = []
        for true_cex_model in true_cex_models:
            # Deepcopy the countermodels so the originals are not affected
            true_cex_offset_model = true_cex_model[0].copy()
            # Make the universe of the model positive and shift the model by accumulated offset
            true_cex_model_universe = true_cex_offset_model.get_fg_elements()
            non_negative_offset = min(true_cex_model_universe)
            if non_negative_offset >= 0:
                non_negative_offset = 0
            true_cex_offset_model.add_fg_element_offset(abs(non_negative_offset) + accumulated_offset)
            # Compute new accumulated offset
            accumulated_offset = max(true_cex_model_universe) + abs(non_negative_offset) + accumulated_offset + 1
            true_cex_model_term_universe = { IntVal(elem) for elem in true_cex_model_universe }
            instantiation_terms = {tuple(elem.as_long() + true_cex_offset_model.offset for elem in instantiation) for instantiation in true_cex_model[1]}
            true_cex_models_with_offset = true_cex_models_with_offset + [(true_cex_offset_model, true_cex_model_term_universe, instantiation_terms)]
        true_cex_models = true_cex_models_with_offset

    all_models = [cex_model.finitemodel for cex_model in cex_models] + [true_cex_model[0].finitemodel for true_cex_model in true_cex_models] + [false_finitemodel.finitemodel]

    vocab = get_vocabulary(annctx)
    set_defs = {func for func in vocab if 'Array' in str(func.range())}
    vocab = vocab.difference(set_defs)

    sygus_model_definitions = sygusBigModelEncoding(all_models, vocab, set_defs, annctx)
    with open(out_file, 'w') as out:
        if options.synthesis_solver == options.minisy:
            out.write('(set-option :smt.random-seed 0)\n')
            out.write('(set-option :sat.random-seed 0)\n')
        if options.synthesis_solver == options.minisy:
            out.write(z3Preamble())
            out.write('\n')
        elif options.synthesis_solver == options.cvc4sy:
            out.write('(set-logic ALL)\n')
        out.write(';; combination of true models and false model\n')
        out.write(sygus_model_definitions)
        out.write('\n\n')
        # Must modify grammar string to include arguments based on problem parameters.
        # Or generate the grammar file based on problem parameters.
        out.write(grammar_string)
        out.write('\n')
        out.write(';; pfp constraints from counterexample models\n')
        if cex_models:
            cex_pfp_constraints = generateAllCexConstraints(cex_models, lemma_args, annctx)
            out.write(cex_pfp_constraints)
            out.write('\n')
        out.write('\n')
        out.write(';; constraints from false model\n')
        false_constraints = generateConstraints(false_finitemodel, lemma_args, goal_instantiation_terms, False, annctx)
        out.write(false_constraints)
        out.write('\n')
        out.write('\n')
        out.write(';; constraints from true counterexample models\n')
        if true_cex_models:
            true_constraints = ''
            for true_cex_model in true_cex_models:
                curr_true_constraint = generateConstraints(true_cex_model[0], lemma_args, true_cex_model[1], True, annctx, instantiations=true_cex_model[2])
                true_constraints += curr_true_constraint + '\n'
            out.write(true_constraints)
        out.write('\n')
        out.write('(check-synth)')
        out.close()
    if options.analytics:
        config_params['analytics']['proposal_start_time'] = time.time()
    # Optionally prefetching a bunch of lemmas to check each one rather than iterating through each one.
    if options.streaming_synthesis_swtich:
        # Default streaming timeout
        streaming_timeout = config_params.get('streaming_timeout', 60)
        k_lemmas_file = '{}/{}_KLemmas.stream'.format(options.log_file_path, problem_instance_name)
        if options.synthesis_solver != options.cvc4sy:
            raise RuntimeError('Streaming only supported wtih CVC4Sy.')
        ps = ProcessStreamer(cmdlist=['cvc4', '--lang=sygus2', '--sygus-stream', out_file],
                             timeout=streaming_timeout,
                             logfile=k_lemmas_file)
        if options.analytics:
            # Analytics are needed. So stream only when next() is called on the generator
            # This will help time the production of each proposal
            return process_synth_results(ps.stream(lazystream=True))
        else:
            return process_synth_results(ps.stream())
    else:
        if options.synthesis_solver == options.minisy:
            proc = subprocess.Popen('minisy {} --smtsolver=z3'.format(out_file),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True, universal_newlines=True)
            minisy_out, err = proc.communicate()
            # Convert output to string
            minisy_out, err = str(minisy_out), str(err)
            if minisy_out == '':
                print(err)
                return None
            return process_synth_results(iter(minisy_out.strip().split('\n')))
        else:
            proc = subprocess.Popen(['cvc4', '--lang=sygus2', out_file],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            cvc4_out, err = proc.communicate()
            if cvc4_out == '':
                print(err)
                return None
            res = cvc4_out.strip().split('\n')
            if res[0] == 'unknown':
                print('Synthesis engine returned unknown.')
                return None
            return process_synth_results(iter(res[1:]))
