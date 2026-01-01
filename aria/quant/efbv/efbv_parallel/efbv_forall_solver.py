"""Forall Solver for EFBV parallel module."""

import logging
from typing import List

import z3

from aria.quant.efbv.efbv_parallel.efbv_checker_utils import parallel_check_candidates
from aria.quant.efbv.efbv_parallel.efbv_utils import FSolverMode

logger = logging.getLogger(__name__)

m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD


class ForAllSolver:
    """Forall solver for EFBV problems."""

    def __init__(self, ctx: z3.Context):
        """Initialize forall solver."""
        # self.forall_vars = []
        self.ctx = ctx  # the Z3 context of the main thread
        # self.phi = None

    def push(self):
        """Push solver state (no-op)."""

    def pop(self):
        """Pop solver state (no-op)."""

    def check(self, cnt_list: List[z3.BoolRef]):
        """Check candidate formulas."""
        if m_forall_solver_strategy == FSolverMode.SEQUENTIAL:
            return self.sequential_check(cnt_list)
        if m_forall_solver_strategy == FSolverMode.PARALLEL_THREAD:
            return self.parallel_check_thread(cnt_list)
        if m_forall_solver_strategy == FSolverMode.PARALLEL_PROCESS:
            return self.parallel_check_process(cnt_list)
        raise NotImplementedError

    def sequential_check(self, cnt_list: List[z3.BoolRef]):
        """Check one-by-one."""
        models = []
        for cnt in cnt_list:
            s = z3.SolverFor("QF_BV")
            s.add(cnt)
            res = s.check()
            if res == z3.sat:
                m = s.model()
                models.append(m)
            elif res == z3.unsat:
                return []  # at least one is UNSAT
        return models

    def parallel_check_thread(self, cnt_list: List[z3.BoolRef]):
        """Solve each formula in cnt_list in parallel."""
        logger.debug("Forall solver: Parallel checking the candidates")
        models_in_other_ctx = parallel_check_candidates(cnt_list, num_workers=4)
        res = []  # translate the model to the main thread
        for m in models_in_other_ctx:
            res.append(m.translate(self.ctx))
        # res = parallel_check_sat_multiprocessing(cnt_list, 4) # this one has bugs
        return res

    def parallel_check_process(self, cnt_list: List[z3.BoolRef]):
        """Parallel check using processes (not implemented)."""
        raise NotImplementedError

    def build_mappings(self):
        """Build the mapping for replacement (not used for now).

        mappings = []
        for v in m:
            mappings.append((z3.BitVec(str(v), v.size(), origin_ctx),
                           z3.BitVecVal(m[v], v.size(), origin_ctx)))
        """
        raise NotImplementedError
