"""
This module contains the code for the CEGIS-based algorithm for EFSMT.
"""

from typing import List
import logging
import z3
from pysmt.logics import QF_BV, QF_LIA, QF_LRA, AUTO
from aria.efmc.smttools.pysmt_solver import PySMTSolver

logger = logging.getLogger(__name__)


def simple_cegis_efsmt(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    logic: str,
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    phi: z3.ExprRef,
    maxloops=None,
    pysmt_solver="z3",
    timeout=None,
):
    """
    A function to solve EFSMT using the CEGIS algorithm
    :param logic: The logic to use for solving
    :param x: The list of existential variables
    :param y: The list of universal variables
    :param phi: The z3 formula to solve
    :param maxloops: The maximum number of loops to run
    :param pysmt_solver: The PySMT solver to use
    :param timeout: The timeout for the solver
    :return: The solution
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
    # FIXME: Note that in pySMT, we need additional work to install Yices2,
    # CVC5, and MathSAT5, so that the users can choose the solver for
    # existential and universal quantifiers
    return sol.efsmt(
        evars=x,
        uvars=y,
        z3fml=phi,
        logic=qf_logic,
        maxloops=maxloops,
        esolver_name=pysmt_solver,
        fsolver_name=pysmt_solver,
        timeout=timeout,
    )
