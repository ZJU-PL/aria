# coding: utf-8
"""
This module provides a MaxSATSolver class that wraps different MaxSAT engines and implements
methods for solving weighted and unweighted MaxSAT problems. It also includes an implementation
of Nadel's algorithm for OMT(BV) "Bit-Vector Optimization (TACAS'16)".
"""

import copy
import time
import logging
from pysat.formula import WCNF
from pysat.solvers import Solver
from dataclasses import dataclass
from typing import List, Optional, Dict, Union


from .fm import FM  # is the FM correct???
from .rc2 import RC2
from .bs import obv_bs, obv_bs_anytime


logger = logging.getLogger(__name__)


@dataclass
class MaxSATSolverResult:
    """Stores the results of a MaxSAT solving operation"""
    cost: float
    solution: Optional[List[int]] = None
    runtime: Optional[float] = None
    status: str = "unknown"
    statistics: Optional[Dict] = None


class MaxSATSolver:
    """
    Wrapper of the engines in maxsat
    """

    maxsat_engine: str
    wcnf: WCNF
    hard: List[List[int]]
    soft: List[List[int]]
    weight: List[Union[int, float]]
    sat_engine_name: str

    def __init__(self, formula: WCNF) -> None:
        """
        :param formula: input MaxSAT formula
        """
        self.maxsat_engine = "FM"
        self.wcnf = formula
        self.hard = copy.deepcopy(formula.hard)
        self.soft = copy.deepcopy(formula.soft)
        self.weight = formula.wght[:]

        self.sat_engine_name = "m22"
        # g3, g4, lgl, mcb, mcm, mpl, m22, mc, mgh, z3

    def set_maxsat_engine(self, name: str) -> None:
        self.maxsat_engine = name

    def get_maxsat_engine(self) -> str:
        """Get MaxSAT engine"""
        return self.maxsat_engine

    def solve_wcnf(self) -> float:
        """TODO: support Popen-based approach for calling bin solvers (e.g., open-wbo)"""
        if self.maxsat_engine == "FM":
            fm = FM(self.wcnf, verbose=0)
            fm.compute()
            # print("cost, ", fm.cost)
            return fm.cost
        elif self.maxsat_engine == "RC2":
            rc2 = RC2(self.wcnf)
            rc2.compute()
            return rc2.cost
        else:
            fm = FM(self.wcnf, verbose=0)
            fm.compute()
            return fm.cost

    def tacas16_binary_search(self) -> Optional[List[int]]:
        """
        Implement Nadel's algorithm for OMT(BV) "Bit-Vector Optimization (TACAS'16)"

        Key idea: OMT on unsigned BV can be seen as lexicographic optimization over the bits in the
        bitwise representation of the objective, ordered from the most-significant bit (MSB)
        to the least-significant bit (LSB).

        Notice that, in this domain, this corresponds to a binary search over the space of the values of the objective

        NOTE: we assume that each element in self.soft is a unary clause, i.e., self.soft is [[l1], [l2], ...]
        """
        bits = []
        for i in reversed(range(len(self.soft))):
            bits.append(self.soft[i][0])

        assumption_lits = []
        try:
            assert self.sat_engine_name != "z3"
            """Use a SAT solver supported by pySAT"""
            sat_oracle = Solver(name=self.sat_engine_name, bootstrap_with=self.hard, use_timer=True)
            # For each bit in bits, decide if it can be true
            for b in bits:
                assumption_lits.append(b)
                if not sat_oracle.solve(assumptions=assumption_lits):
                    # if b cannot be positive, then set it to be negative?
                    # after this round, we will try the remaining bits
                    assumption_lits.pop()
                    assumption_lits.append(-b)
        except Exception as ex:
            print(ex)
        # print("final assumptions: ", assumption_lits)
        return assumption_lits

        # return obv_bs(self.hard, len(self.soft))  # FIXME: @wwq, it seems that this function has bugs

    def solve(self) -> MaxSATSolverResult:
        """
        Solve the MaxSAT problem using the selected engine

        Returns:
            SolverResult containing cost, solution, runtime and statistics
        """
        start_time = time.time()
        result = MaxSATSolverResult(float('inf'))

        try:
            if self.maxsat_engine == "FM":
                fm = FM(self.wcnf, verbose=0)
                found_solution = fm.compute()
                result.cost = fm.cost
                result.solution = getattr(fm, 'model', None)
                # FM algorithm finds optimum when it completes successfully
                result.status = "optimal" if found_solution and hasattr(fm, 'model') and fm.model is not None else "satisfied"

            elif self.maxsat_engine == "RC2":
                rc2 = RC2(self.wcnf)
                model = rc2.compute()
                result.cost = rc2.cost
                result.solution = model
                result.status = "optimal" if model is not None else "unknown"

            elif self.maxsat_engine == "OBV-BS":
                bits = [self.soft[i][0] for i in reversed(range(len(self.soft)))]
                solution = obv_bs(self.hard, bits)
                result.cost = sum(1 for bit in solution if bit > 0)
                result.solution = solution
                result.status = "optimal"

            elif self.maxsat_engine == "OBV-BS-ANYTIME":
                bits = [self.soft[i][0] for i in reversed(range(len(self.soft)))]
                solution = obv_bs_anytime(self.hard, bits, time_limit=self.timeout)
                result.cost = sum(1 for bit in solution if bit > 0)
                result.solution = solution
                result.status = "optimal" if len(solution) == len(bits) else "timeout"

            else:
                logger.warning(f"Unknown engine {self.maxsat_engine}, using FM")
                fm = FM(self.wcnf, verbose=0)
                found_solution = fm.compute()
                result.cost = fm.cost
                result.solution = getattr(fm, 'model', None)
                # FM algorithm finds optimum when it completes successfully
                result.status = "optimal" if found_solution and hasattr(fm, 'model') and fm.model is not None else "satisfied"

        except Exception as e:
            logger.error(f"Error solving MaxSAT problem: {str(e)}")
            result.status = "error"
            result.statistics = {"error": str(e)}

        finally:
            result.runtime = time.time() - start_time


        return result
