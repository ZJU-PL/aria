from z3 import *
from aria.quant.fossil.lemsynth.utils import *
import itertools
import random


# Each true model is a dictionary with keys being names of constants/functions,
# and values being the valuation of the item. For a constant, the entry is just
# the value of the constant. For a function, the value is a dictionary from
# tuples of arguments to the corresponding outputs.

# TODO: consider adding the signature of the constant/function to the key of the model


# Generates a list of dictionaries with key corresponding to the function name
# and valuation be ng one of many possible valuations
def getTrueModelFctValuations(elems, fct_signature, fct_name):
    arity = fct_signature[0]
    if arity == 0:
        # dict list evaluating the constant symbol to any of the elements
        return [{fct_name: elem} for elem in elems]
    else:
        # Only supporting integer type, with many arguments and one output
        # Return list of dictionaries evaluating each input element/tuples of
        # elements to aribtrary elements
        if arity == 1:
            input_values = elems
        else:
            input_values = [tuple(x) for x in itertools.product(elems, repeat=arity)]
        domain_size = len(input_values)
        output_valuations = [
            list(x) for x in itertools.product(elems, repeat=domain_size)
        ]
        # Writing in a loop and not in a list comprehension to make it easier to understand
        # [{input_values[i] : x[i] for i in range(domain_size)} for x in output_valuations]
        result = []
        for i in range(len(output_valuations)):
            one_possible_valuation = {
                input_values[k]: output_valuations[i][k] for k in range(domain_size)
            }
            result = result + [{fct_name: one_possible_valuation}]
        return result


# Generate true models in the form of all posible evaluations of all functions
# This generates all possible combinations. Maybe better to get a few random
# ones instead if this is slow
def getTrueModels(elems, fcts_z3):
    models = [{"elems": elems}]
    for key in fcts_z3.keys():
        fct_signature = getFctSignature(key)
        if fct_signature[3] == True:
            # Recursive function. No need to find valuation. Continue.
            continue
        else:
            for fct in fcts_z3[key]:
                fct_name = getZ3FctName(fct)
                submodel_fct = getTrueModelFctValuations(elems, fct_signature, fct_name)
                models = listProduct(models, submodel_fct, lambda x, y: {**x, **y})
    return models


def getRandTrueModelFctValuations(elems, fct_signature, num_valuations):
    fct_valuations = []
    arity = fct_signature[0]
    if arity == 0:
        # Possible values for a constant. Return num_valuations many choices.
        return random.choices(elems, k=num_valuations)
    else:
        # Only supporting integer type, with many arguments and one output
        if arity == 1:
            input_values = elems
        else:
            input_values = [tuple(x) for x in itertools.product(elems, repeat=arity)]
        domain_size = len(input_values)
        for i in range(num_valuations):
            output_valuation = random.choices(elems, k=domain_size)
            fct_valuation = {
                input_values[j]: output_valuation[j] for j in range(domain_size)
            }
            fct_valuations = fct_valuations + [fct_valuation]
        return fct_valuations


def getRandTrueModels(elems, fcts_z3, num_true_models):
    models = []
    # Loop over number of true models. Might be useful to add some kind of equality check here to get distinct models, but that could be expensive.
    for i in range(num_true_models):
        model = {"elems": elems}
        for key in fcts_z3.keys():
            fct_signature = getFctSignature(key)
            if fct_signature[3] == True:
                # Recursive function. No need to find valuation. Continue.
                continue
            elif fct_signature[2] != None and fct_signature[2] != "int":
                # Return type not an integer. Continue.
                # TODO: VERY IMPORTANT: this could affect distinguishing by sorts.
                continue
            else:
                for fct in fcts_z3[key]:
                    fct_name = getZ3FctName(fct)
                    fct_valuations = getRandTrueModelFctValuations(
                        elems, fct_signature, 1
                    )
                    model[fct_name] = fct_valuations[0]
        models = models + [model]
    return models


# Initialize dictionary where given recdef's name is evaluated to a dictionary
## where all elements have the initial value (lattice bottom)
def initializeRecdef(model, recdef, key):
    # only supporting boolean recursive predicates. Initial value is false
    elems = model["elems"]
    recdef_name = getRecdefName(recdef)
    bottom_elt = getBottomElement(key)
    init = {recdef_name: {elem: bottom_elt for elem in elems}}
    return init


# evaluate model via given unfolded recdef function until fixpoint is reached
# all recdefs assumed unary predicates
# TODO: mutually recursive definitions
def evaluateUntilFixpoint(recdef_lookup, model, prev_model={}):
    if model == prev_model:
        return model
    else:
        recdef_names = recdef_lookup.keys()
        new_prev = deepcopyModel(model)
        elems = model["elems"]
        for elem in elems:
            for recdef_name in recdef_names:
                recdef_function = recdef_lookup[recdef_name]
                new_val = recdef_function(elem, model)
                model[recdef_name][elem] = new_val
        return evaluateUntilFixpoint(recdef_lookup, model, new_prev)


# Alternate definition of filter by python axioms where axioms are distinguished
# by signature
def filterByAxiomsFct(model, axioms_python):
    for axiom_class in axioms_python.keys():
        signature = getFctSignature(axiom_class)
        arity = signature[0]
        if arity == 0:
            for axiom in axioms_python[axiom_class]:
                if not axiom(model):
                    return False
        else:
            elems = model["elems"]
            if arity == 1:
                input_values = elems
            else:
                input_values = [
                    tuple(x) for x in itertools.product(elems, repeat=arity)
                ]
            for axiom in axioms_python[axiom_class]:
                for input_value in input_values:
                    if not axiom(input_value, model):
                        return False
    # Default case. All axioms are satisfied
    return True


# Function to evaluate recursive definitions on a given true model
def getRecdefsEval(model, unfold_recdefs_python):
    recdef_lookup = {}
    for key in unfold_recdefs_python.keys():
        recdefs = unfold_recdefs_python[key]
        # Lookup must eventually be distinguished by signature
        recdef_lookup.update({getRecdefName(recdef): recdef for recdef in recdefs})
        for recdef in recdefs:
            init_rec = initializeRecdef(model, recdef, key)
            model.update(init_rec)
    # Evaluate recursive definitions
    eval_model = evaluateUntilFixpoint(recdef_lookup, model)
    return eval_model


# Get true models (with offsets added) with recdef evaluations such that they
# satisfy axioms.
def getNTrueModels(
    elems,
    fcts_z3,
    unfold_recdefs_python,
    axioms_python,
    true_model_offset,
    config_params,
):
    # Base model either gotten through complete enumeration or at random
    mode = config_params.get("mode", None)
    num_true_models = config_params.get("num_true_models", 0)
    if num_true_models == 0:
        return []
    elif mode is None:
        raise ValueError(
            "Must specify true models to be generated in either enumeration mode or random mode."
        )
    elif mode == "random":
        num_true_models = 1 if not isinstance(num_true_models, int) else num_true_models
        search_fuel = abs(config_params.get("fuel", 3))
    elif mode == "enumeration":
        search_fuel = 1
    else:
        raise ValueError(
            "Incorrect mode: Must specify true models to be generated in either enumeration mode or random mode."
        )

    evaluated_models = []
    filtered_models = []
    while (
        not isinstance(num_true_models, int) or len(filtered_models) < num_true_models
    ) and search_fuel > 0:
        if mode == "enumeration":
            true_models_base = getTrueModels(elems, fcts_z3)
        elif mode == "random":
            true_models_base = getRandTrueModels(elems, fcts_z3, num_true_models)
        else:
            raise ValueError("Cannot understand mode for true model generation.")
        # Evaluate recdefs
        for model_base in true_models_base:
            evaluated_models += [getRecdefsEval(model_base, unfold_recdefs_python)]
        # Filter by axioms
        for model in evaluated_models:
            pass_or_fail = filterByAxiomsFct(model, axioms_python)
            if pass_or_fail:
                filtered_models = filtered_models + [model]
        # Decrease fuel and search again if enough models not obtained. In enumeration mode this exits the loop after one pass.
        search_fuel = search_fuel - 1

    # Post-processing
    # First, extract the desired number of models.
    # For random mode this could be expensive to do again. Must add code to skip in random case.
    if num_true_models == "full" or (
        isinstance(num_true_models, int) and num_true_models > len(filtered_models)
    ):
        choice_models = filtered_models
    elif isinstance(num_true_models, int) and num_true_models <= len(filtered_models):
        choice_models = random.choices(filtered_models, k=num_true_models)
    else:
        raise ValueError("Must specify either a number of models or 'full'")

    # Second, make all the elements of all the models positive and add offsets so that they have distinct universes.
    final_models = []
    accumulated_offset = true_model_offset
    for choice_model in choice_models:
        # Make the universe of the true model positive
        choice_model_positive_universe = makeModelUniverseNonNegative(choice_model)
        # Shift the model by accumulated offset
        final_model = addOffset(
            choice_model_positive_universe, lambda x: accumulated_offset + 10 + x
        )
        # Compute new accumulated offset and accumulate
        accumulated_offset = getRelativeModelOffset(final_model)
        # Add model to final_models
        final_models = final_models + [final_model]
    return final_models
