"""CEGIS-based solvers for EFBV problems."""

from typing import List
import logging
import z3
from aria.utils.pysmt_solver import PySMTSolver
from pysmt.logics import QF_BV, QF_LIA, QF_LRA, AUTO

logger = logging.getLogger(__name__)


def simple_cegis_efsmt(
    logic: str,
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    phi: z3.ExprRef,
    maxloops=None,
    *,
    pysmt_solver="z3",
):
    """
    Solve EFSMT using the CEGIS algorithm.

    Args:
        logic: The logic to use for solving
        x: The list of existential variables
        y: The list of universal variables
        phi: The z3 formula to solve
        maxloops: The maximum number of loops to run
        pysmt_solver: The pysmt solver to use

    Returns:
        The solution
    """
    if "IA" in logic:
        qf_logic = QF_LIA
    elif "RA" in logic:
        qf_logic = QF_LRA
    elif "BV" in logic:
        qf_logic = QF_BV
    else:
        qf_logic = AUTO
    sol = PySMTSolver()
    return sol.efsmt(
        evars=x,
        uvars=y,
        z3fml=phi,
        logic=qf_logic,
        maxloops=maxloops,
        esolver_name=pysmt_solver,
        fsolver_name=pysmt_solver,
    )
