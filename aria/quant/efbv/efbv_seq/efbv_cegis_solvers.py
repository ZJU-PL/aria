from typing import List
import logging
import z3
from aria.utils.pysmt_solver import PySMTSolver

logger = logging.getLogger(__name__)


def simple_cegis_efsmt(logic: str, x: List[z3.ExprRef], y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None,
                       profiling=False, pysmt_solver="z3"):
    """
    A function to solve EFSMT using the CEGIS algorithm
    :param logic: The logic to use for solving
    :param x: The list of existential variables
    :param y: The list of universal variables
    :param phi: The z3 formula to solve
    :param maxloops: The maximum number of loops to run
    :param profiling: Whether to enable profiling or not
    :return: The solution
    """
    from pysmt.logics import QF_BV, QF_LIA, QF_LRA, AUTO
    if "IA" in logic:
        qf_logic = QF_LIA
    elif "RA" in logic:
        qf_logic = QF_LRA
    elif "BV" in logic:
        qf_logic = QF_BV
    else:
        qf_logic = AUTO
    sol = PySMTSolver()
    return sol.efsmt(evars=x, uvars=y, z3fml=phi,
                     logic=qf_logic, maxloops=maxloops,
                     esolver_name=pysmt_solver, fsolver_name=pysmt_solver)
