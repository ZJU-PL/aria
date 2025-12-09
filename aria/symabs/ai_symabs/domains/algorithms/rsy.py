"""Implementation of the RSY algorithm for calculating alpha-tilde.
Can also be used to calculate post-tilde.
"""
from typing import Any
from .timeout import TimeoutException


# pylint: disable=invalid-name
def RSY(domain: Any, phi: Any) -> Any:
    """Returns alpha-tilde of phi.
    If the SMT solver never times out and the function runs to completion, it
    will return alpha-hat. Otherwise, it will return the trivial
    overapproximation of domain.top.
    See Algorithm 6 in:
    Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
    Applications (Ph.D. dissertation). Computer Sciences Department, University
    of Wisconsin, Madison.
    """
    lower = domain.bottom

    while True:
        try:
            S = domain.model_and(phi,
                                 domain.logic_not(domain.gamma_hat(lower)))
        except TimeoutException:
            return domain.top

        if S is None:
            break
        else:
            lower = domain.join([lower, domain.beta(S)])

    return lower
