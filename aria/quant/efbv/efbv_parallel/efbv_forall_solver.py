"""Forall Solver for EFBV parallel module."""

import logging
import concurrent.futures
from typing import List

import z3
from aria.quant.efbv.efbv_parallel.efbv_utils import FSolverMode
from aria.quant.efbv.efbv_parallel.exceptions import (
    ForAllSolverSuccess,
    ForAllSolverUnknown,
)

logger = logging.getLogger(__name__)

m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD


class ForAllSolver:
    """Forall solver for EFBV problems."""

    def __init__(self, ctx: z3.Context, num_workers: int = 4):
        """Initialize forall solver."""
        # self.forall_vars = []
        self.ctx = ctx  # the Z3 context of the main thread
        # self.phi = None
        self.num_workers = num_workers
        self._worker_solvers = []
        self._worker_contexts = []

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
        solver = z3.SolverFor("QF_BV", ctx=self.ctx)
        for cnt in cnt_list:
            solver.push()
            solver.add(cnt)
            try:
                res = solver.check()
                if res == z3.sat:
                    models.append(solver.model())
                elif res == z3.unsat:
                    raise ForAllSolverSuccess()
                else:
                    raise ForAllSolverUnknown()
            finally:
                solver.pop()
        return models

    def _ensure_worker_pool(self):
        if self._worker_solvers:
            return
        for _ in range(self.num_workers):
            worker_ctx = z3.Context()
            self._worker_contexts.append(worker_ctx)
            self._worker_solvers.append(z3.SolverFor("QF_BV", ctx=worker_ctx))

    def _check_in_worker(self, worker_idx: int, cnt: z3.BoolRef) -> z3.ModelRef:
        solver = self._worker_solvers[worker_idx]
        worker_ctx = self._worker_contexts[worker_idx]
        local_cnt = cnt.translate(worker_ctx)
        solver.push()
        try:
            solver.add(local_cnt)
            res = solver.check()
            if res == z3.sat:
                return solver.model()
            if res == z3.unsat:
                raise ForAllSolverSuccess()
            raise ForAllSolverUnknown()
        finally:
            solver.pop()

    def parallel_check_thread(self, cnt_list: List[z3.BoolRef]):
        """Solve each formula in cnt_list in parallel."""
        logger.debug("Forall solver: Parallel checking the candidates")
        self._ensure_worker_pool()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = []
            for idx, cnt in enumerate(cnt_list):
                worker_idx = idx % self.num_workers
                futures.append(
                    executor.submit(self._check_in_worker, worker_idx, cnt)
                )
            models_in_other_ctx = [f.result() for f in futures]
        return [m.translate(self.ctx) for m in models_in_other_ctx]

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
