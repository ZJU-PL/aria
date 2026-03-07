"""Forall Solver for EFBV parallel module."""

import logging
import concurrent.futures
from typing import Dict, List

import z3
from aria.quant.efbv.efbv_parallel.efbv_utils import FSolverMode
from aria.quant.efbv.efbv_parallel.exceptions import (
    ForAllSolverSuccess,
    ForAllSolverUnknown,
)

logger = logging.getLogger(__name__)

m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD


class ModelSnapshot:
    """Minimal model wrapper for cross-context variable evaluation."""

    def __init__(self, assignments: Dict[str, z3.ExprRef]):
        self.assignments = assignments

    def eval(self, expr: z3.ExprRef, model_completion: bool = False) -> z3.ExprRef:
        del model_completion
        if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            value = self.assignments.get(str(expr))
            if value is not None:
                return value
        raise KeyError(f"No assignment recorded for expression: {expr}")


class ForAllSolver:
    """Forall solver for EFBV problems."""

    def __init__(self, ctx: z3.Context, num_workers: int = 4):
        """Initialize forall solver."""
        # self.forall_vars = []
        self.ctx = ctx  # the Z3 context of the main thread
        # self.phi = None
        self.num_workers = num_workers

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
        # No longer needed - we create a new solver for each task
        pass

    def _serialize_model(
        self, model: z3.ModelRef, expr: z3.ExprRef
    ) -> Dict[str, z3.ExprRef]:
        assignments: Dict[str, z3.ExprRef] = {}
        stack = [expr]
        seen = set()
        while stack:
            current = stack.pop()
            key = current.get_id()
            if key in seen:
                continue
            seen.add(key)
            if z3.is_const(current) and current.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                assignments[str(current)] = model.eval(current, model_completion=True)
            stack.extend(current.children())
        return assignments

    def _check_in_worker(self, worker_idx: int, cnt: z3.BoolRef) -> Dict[str, z3.ExprRef]:
        # Create a new context and solver for each task to avoid thread-safety issues
        # Z3 solvers are not thread-safe, so we cannot reuse solver instances
        del worker_idx
        worker_ctx = z3.Context()
        local_cnt = cnt.translate(worker_ctx)
        solver = z3.SolverFor("QF_BV", ctx=worker_ctx)
        solver.add(local_cnt)
        res = solver.check()
        if res == z3.sat:
            return self._serialize_model(solver.model(), local_cnt)
        if res == z3.unsat:
            raise ForAllSolverSuccess()
        raise ForAllSolverUnknown()

    def parallel_check_thread(self, cnt_list: List[z3.BoolRef]):
        """Solve each formula in cnt_list in parallel."""
        logger.debug("Forall solver: Parallel checking the candidates")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = []
            for cnt in cnt_list:
                # worker_idx is not used anymore but kept for compatibility
                futures.append(
                    executor.submit(self._check_in_worker, 0, cnt)
                )
            assignment_sets = [f.result() for f in futures]
        translated = []
        for assignments in assignment_sets:
            translated.append(
                ModelSnapshot(
                    {
                        name: value.translate(self.ctx)
                        for name, value in assignments.items()
                    }
                )
            )
        return translated

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
