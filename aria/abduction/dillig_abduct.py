"""
Dillig-style abduction implementation.

The key idea is to find a minimal satisfying assignment (MSA) that makes
the formula pre_cond -> post_cond valid, and then generalize it through
quantifier elimination.
"""

import z3
from aria.utils.z3_solver_utils import is_sat, is_entail
from aria.utils.z3_expr_utils import get_variables
from aria.optimization.msa.mistral_msa import MSASolver


def generalize_model(model, pre_cond, post_cond):
    """
    Generalize a model through quantifier elimination.

    Args:
        model: Model from MSA solver
        pre_cond: Precondition formula
        post_cond: Postcondition formula

    Returns:
        Generalized formula
    """
    # Extract model constraints
    model_constraints = []
    for var, value in model.items():
        model_constraints.append(var == value)

    if not model_constraints:
        # If no constraints, try using the postcondition directly
        return post_cond

    # Create a formula representing the model
    model_formula = z3.And(*model_constraints)

    # Perform quantifier elimination to generalize the model
    # Variables to keep: those in post_cond but not in pre_cond
    post_vars = set(get_variables(post_cond))
    pre_vars = set(get_variables(pre_cond))
    vars_to_keep = post_vars - pre_vars

    # Variables to eliminate: all others in the model
    model_vars = set(get_variables(model_formula))
    vars_to_forget = model_vars - vars_to_keep

    if not vars_to_forget:
        # Check if model_formula is sufficient
        if is_entail(z3.And(pre_cond, model_formula), post_cond):
            return model_formula
        # If not sufficient, use post_cond directly
        return post_cond

    # Try direct quantifier elimination on the model formula
    if vars_to_forget:
        try:
            # First approach: eliminate variables from model_formula
            qfml = z3.ForAll(list(vars_to_forget), model_formula)
            qe_result = z3.Tactic("qe2").apply(qfml)

            if qe_result and len(qe_result) > 0:
                result = qe_result[0].as_expr()
                # Check if the result is sufficient
                if is_entail(z3.And(pre_cond, result), post_cond):
                    return result

            # Second approach: eliminate variables from model_formula -> post_cond
            qfml = z3.ForAll(
                list(vars_to_forget), z3.Implies(model_formula, post_cond)
            )
            qe_result = z3.Tactic("qe2").apply(qfml)

            if qe_result and len(qe_result) > 0:
                result = qe_result[0].as_expr()
                # Check if the result is sufficient
                if is_entail(z3.And(pre_cond, result), post_cond):
                    return result
        except z3.Z3Exception as e:
            print(f"QE in Dillig abduction failed: {e}")

    # If all else fails, use the postcondition directly as the abduction
    # This ensures sufficiency but might not be minimal
    return post_cond


def dillig_abduce(pre_cond, post_cond):
    """
    Perform abduction using the Dillig approach.

    This approach finds a minimal satisfying assignment (MSA) for the formula
    pre_cond -> post_cond, and then generalizes it through quantifier elimination.

    Args:
        pre_cond: Precondition formula
        post_cond: Postcondition formula

    Returns:
        The abduced formula, or None if abduction fails
    """
    # Check if the precondition is unsatisfiable
    if not is_sat(pre_cond):
        return z3.BoolVal(True)  # Any formula would work

    # Check if the implication is valid
    if is_entail(pre_cond, post_cond):
        return z3.BoolVal(True)  # No additional constraints needed

    # Check if the implication is unsatisfiable
    if not is_sat(z3.And(pre_cond, post_cond)):
        # If pre_cond and post_cond are mutually exclusive, no abduction is possible
        return None

    # Create the formula for which we need to find an MSA
    formula = z3.Implies(pre_cond, post_cond)

    # Initialize the MSA solver
    msa = MSASolver()
    msa.init_from_formula(formula)

    # Find a minimal satisfying assignment
    model = msa.find_small_model()
    if model is False:  # MSASolver returns False if no model is found
        return None

    # Extract model constraints
    model_constraints = []
    for var in get_variables(formula):
        if var.decl() in model.decls():
            model_constraints.append(var == model[var])

    if not model_constraints:
        # If no constraints, try using the postcondition directly
        return post_cond

    # Create a formula representing the model
    model_formula = z3.And(*model_constraints)

    # Check if the model formula is consistent with the precondition
    if not is_sat(z3.And(pre_cond, model_formula)):
        # If not consistent, use the postcondition directly
        return post_cond

    # Check if the model formula is sufficient
    if not is_entail(z3.And(pre_cond, model_formula), post_cond):
        # If not sufficient, use the postcondition directly
        return post_cond

    # Perform quantifier elimination to generalize the model
    # Variables to keep: those in post_cond but not in pre_cond
    post_vars = set(get_variables(post_cond))
    pre_vars = set(get_variables(pre_cond))
    vars_to_keep = post_vars - pre_vars

    # Variables to eliminate: all others in the model
    model_vars = set(get_variables(model_formula))
    vars_to_forget = model_vars - vars_to_keep

    if not vars_to_forget:
        return model_formula

    # Try direct quantifier elimination on the model formula
    if vars_to_forget:
        try:
            # First approach: eliminate variables from model_formula
            qfml = z3.ForAll(list(vars_to_forget), model_formula)
            qe_result = z3.Tactic("qe2").apply(qfml)

            if qe_result and len(qe_result) > 0:
                result = qe_result[0].as_expr()
                # Check if the result is consistent and sufficient
                if (is_sat(z3.And(pre_cond, result)) and
                        is_entail(z3.And(pre_cond, result), post_cond)):
                    return result

            # Second approach: eliminate variables from model_formula -> post_cond
            qfml = z3.ForAll(
                list(vars_to_forget), z3.Implies(model_formula, post_cond)
            )
            qe_result = z3.Tactic("qe2").apply(qfml)

            if qe_result and len(qe_result) > 0:
                result = qe_result[0].as_expr()
                # Check if the result is consistent and sufficient
                if (is_sat(z3.And(pre_cond, result)) and
                        is_entail(z3.And(pre_cond, result), post_cond)):
                    return result
        except z3.Z3Exception as e:
            print(f"QE in Dillig abduction failed: {e}")

    return None
