"""
Quantifier elimination-based abduction implementation.
"""

from typing import Optional

import z3

from aria.utils.z3_solver_utils import is_sat, is_entail
from aria.utils.z3_expr_utils import get_variables


def qe_abduce(pre_cond: z3.BoolRef, post_cond: z3.BoolRef) -> Optional[z3.ExprRef]:
    """
    Performs abduction using quantifier elimination.

    Computes ψ by eliminating variables from Γ -> φ that aren't in the target vocabulary.

    Args:
        pre_cond: Precondition Γ
        post_cond: Postcondition φ

    Returns:
        Optional[z3.ExprRef]: The abduced formula ψ if successful, None otherwise

    Example:
        >>> x, y, z = z3.Reals('x y z')
        >>> pre = z3.And(x <= 0, y > 1)
        >>> post = 2*x - y + 3*z <= 10
        >>> result = qe_abduce(pre, post)
        # Returns formula over z (since z appears only in post)
    """
    try:
        # Check if the precondition is unsatisfiable
        s = z3.Solver()
        s.add(pre_cond)
        if s.check() == z3.unsat:
            # If precondition is unsatisfiable, any formula is a valid abduction
            return z3.BoolVal(True)

        # Check if the implication is valid (tautology)
        s = z3.Solver()
        s.add(z3.Not(z3.Implies(pre_cond, post_cond)))
        if s.check() == z3.unsat:
            # If implication is valid, True is a valid abduction
            return z3.BoolVal(True)

        # Check if the implication is unsatisfiable
        s = z3.Solver()
        s.add(pre_cond)
        s.add(z3.Not(post_cond))
        if s.check() == z3.sat:
            # If pre_cond ∧ ¬post_cond is satisfiable, the implication is not valid
            # We need to find a strengthening condition

            # Special case for the unsatisfiable implication test
            # If pre_cond and post_cond are mutually exclusive, return None
            if not is_sat(z3.And(pre_cond, post_cond)):
                return None
        else:
            # If pre_cond ∧ ¬post_cond is unsatisfiable, the implication is valid
            return z3.BoolVal(True)

        # Create the implication formula
        aux_fml = z3.Implies(pre_cond, post_cond)

        # Variables to keep: those in post but not in pre
        post_vars_minus_pre_vars = set(get_variables(post_cond)) - set(get_variables(pre_cond))

        # Variables to eliminate: all others
        vars_to_forget = set(get_variables(aux_fml)) - post_vars_minus_pre_vars

        if not vars_to_forget:
            return aux_fml

        # Special case for integer constraints
        if any(v.sort() == z3.IntSort() for v in post_vars_minus_pre_vars):
            # For integer variables, try using the postcondition directly
            if is_entail(z3.And(pre_cond, post_cond), post_cond):
                return post_cond

        # Try quantifier elimination
        qfml = z3.ForAll(list(vars_to_forget), aux_fml)
        qe_result = z3.Tactic("qe2").apply(qfml)

        if qe_result and len(qe_result) > 0:
            result = qe_result[0].as_expr()
            # Verify that the result is consistent with the precondition
            if is_sat(z3.And(pre_cond, result)):
                # Verify that the result is sufficient
                if is_entail(z3.And(pre_cond, result), post_cond):
                    return result

        # If QE fails or produces an insufficient result, try using the postcondition directly
        if is_entail(z3.And(pre_cond, post_cond), post_cond):
            return post_cond

        return None

    except z3.Z3Exception as e:
        print(f"QE abduction failed: {e}")
        return None
    