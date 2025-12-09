from typing import Dict, Tuple

from z3 import *


def z3_call(formula: str, timeout: int) -> Tuple[bool, Dict[str, float]]:
    """
    Check if the formula is satisfiable using z3 with a timeout.

    Parameters
    ----------
    formula : str
        The formula in smt2 to be checked.
    timeout : int
        The timeout in seconds.

    Returns
    -------
    bool
        The result of the check. True if satisfiable, False if unsatisfiable, None if unknown.
    Dict[str, float]
        The model (if satisfiable).
    """
    s = Solver()
    s.set("timeout", timeout * 1000)
    s.add(parse_smt2_string(formula))

    result = s.check()
    if result == sat:
        model = s.model()
        return True, {d.name(): model[d] for d in model.decls()}
    elif result == unsat:
        return False, {}
    else:
        return None, {}
