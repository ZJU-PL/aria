"""
Solving exists-forall problem over Boolean formulas, sequentially
"""

import logging
from typing import List, Optional

from aria.quant.efbool.efbool_utils import EFBoolResult
from aria.quant.efbool.efbool_exists_solver import BoolExistsSolver
from aria.quant.efbool.efbool_forall_solver import BoolForallSolver

logger = logging.getLogger(__name__)


def solve_ef_bool(
    x: List[int], y: List[int], phi: List[List[int]], maxloops: Optional[int] = None
) -> EFBoolResult:
    """Solving exists-forall problem over Boolean formulas
    :param x: the set of existential quantified variables
    :param y: the set of universal quantified variables
    :param phi: the Boolean formula
    :param maxloops: the maximum number of iterations
    :return:
    """
    esolver = BoolExistsSolver(x, phi)
    fsolver = BoolForallSolver(x, y, phi)
    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("Round: ", loops)
        e_models = esolver.get_models(1)
        if len(e_models) == 0:
            return EFBoolResult.UNSAT
        print(e_models)
        blocking_clauses = fsolver.check_models(e_models)
        if len(blocking_clauses) == 0:
            return EFBoolResult.SAT
        for cls in blocking_clauses:
            esolver.add_clause(cls)
    return EFBoolResult.UNKNOWN
