"""Implementation of the Bilateral algorithm for calculating alpha-tilde.
Can also be used to calculate post-tilde.
"""
from typing import Any, Optional, Callable
from .timeout import TimeoutException


def bilateral(domain: Any, phi: Any,
              descend_check: Optional[int] = None,
              initial_lower: Optional[Any] = None,
              initial_upper: Optional[Any] = None) -> Any:
    """Returns alpha-tilde of phi using the bilateral algorithm.
    If the SMT solver never times out, the function never hits a resource
    limit, and the function runs to completion, it will return alpha-hat.
    Otherwise, it will return a non-trivial overapproximation.
    After descend_check number of meets in a row, it will use p = lower.
    See Algorithm 13 in:
    Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
    Applications (Ph.D. dissertation). Computer Sciences Department, University
    of Wisconsin, Madison.
    """
    upper = initial_upper or domain.top
    lower = initial_lower or domain.bottom

    meets_in_a_row = 0
    while lower != upper:
        if meets_in_a_row is descend_check:
            # We want to break up a possibly-infinite descending chain, so we
            # let p = lower. Thus, if S = None, we'll have upper = meet(upper,
            # lower) = lower and we'll terminate. If S != None, we'll have
            # lower = join(lower, beta(S)), which will cause lower to ascend.
            consequence = lower
        else:
            # Otherwise, let's use the actual abstract consequence
            consequence = domain.abstract_consequence(lower, upper)

        try:
            model = domain.model_and(
                phi, domain.logic_not(domain.gamma_hat(consequence)))
        except TimeoutException:
            return upper

        if model is None:
            new_upper = domain.meet([upper, consequence])
            upper = new_upper
            meets_in_a_row += 1
        else:
            new_lower = domain.join([lower, domain.beta(model)])
            lower = new_lower
            meets_in_a_row = 0

    return upper
