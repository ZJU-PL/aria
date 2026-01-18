"""Parallel CEGIS solver for exists-forall linear integer/real arithmetic."""

from __future__ import annotations

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

import z3
from z3.z3util import get_vars

from aria.global_params.paths import global_config
from aria.quant.eflira.eflira_sampling_utils import (
    ESolverSampleStrategy,
    SamplingUnknown,
    sample_models,
)
from aria.utils import SolverResult
from aria.utils.smtlib_solver import SMTLIBSolver
from aria.utils.exceptions import SMTSuccess, SMTUnknown

logger = logging.getLogger(__name__)


class ExistsSolverSuccess(SMTSuccess):
    """The Exists solver proved UNSAT."""


class ForAllSolverSuccess(SMTSuccess):
    """The Forall solver validated a candidate."""


class ExistsSolverUnknown(SMTUnknown):
    """The Exists solver returned UNKNOWN."""


class ForAllSolverUnknown(SMTUnknown):
    """The Forall solver returned UNKNOWN."""


class EFLIRAResult(Enum):
    """Result of EFLIRA checking."""

    UNSAT = 0
    SAT = 1
    UNKNOWN = 2
    ERROR = 3


class EFLIRASolver(ABC):
    """Abstract base class for EFLIRA solvers."""

    @abstractmethod
    def solve_efsmt_lira(
        self, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef], phi
    ) -> EFLIRAResult:
        """Solve EFLIRA problem (abstract method)."""


class FSolverMode(Enum):
    SEQUENTIAL = 0
    PARALLEL_THREAD = 1
    PARALLEL_PROCESS_IPC = 2


m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD
g_forall_bin_solver = "z3"
g_forall_num_workers = 4


def _infer_qf_logic(vars_list: List[z3.ExprRef]) -> str:
    has_int = any(v.sort().kind() == z3.Z3_INT_SORT for v in vars_list)
    has_real = any(v.sort().kind() == z3.Z3_REAL_SORT for v in vars_list)
    if has_int and has_real:
        return "QF_LIRA"
    if has_real:
        return "QF_LRA"
    return "QF_LIA"


def _make_solver(logic: str, ctx: Optional[z3.Context] = None) -> z3.Solver:
    try:
        return z3.SolverFor(logic, ctx=ctx)
    except z3.Z3Exception:
        return z3.Solver(ctx=ctx)


def _build_smt2_script(logic: str, fml: z3.BoolRef) -> str:
    sol = z3.Solver()
    sol.add(fml)
    return f"(set-logic {logic})\n" + sol.to_smt2()


def _get_solver_cmd(solver_name: str) -> str:
    solver_path = global_config.get_solver_path(solver_name)
    if solver_path is None:
        raise ValueError(f"Binary solver not found: {solver_name}")
    if solver_name == "z3":
        return f"{solver_path} -in"
    if solver_name == "cvc5":
        return f"{solver_path} -q -i --produce-models"
    if solver_name == "yices2":
        return f"{solver_path} --incremental"
    return f"{solver_path} -in"


def _split_value_pairs(raw: str) -> List[str]:
    s = raw.strip()
    if not s.startswith("(") or not s.endswith(")"):
        return []
    s = s[1:-1].strip()
    pairs = []
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                pairs.append(s[start : i + 1])
                start = None
    return pairs


def _parse_values_output(raw: str) -> dict:
    pairs = _split_value_pairs(raw)
    values = {}
    for pair in pairs:
        body = pair.strip()[1:-1].strip()
        if not body:
            continue
        if body.startswith("|"):
            end_idx = body.find("|", 1)
            if end_idx == -1:
                continue
            name = body[: end_idx + 1].strip()
            value = body[end_idx + 1 :].strip()
        else:
            split_at = body.find(" ")
            if split_at == -1:
                continue
            name = body[:split_at].strip()
            value = body[split_at + 1 :].strip()
        values[name] = value
    return values


def _value_from_smt2_string(
    name: str, value_str: str, sort: z3.SortRef, ctx: z3.Context
) -> z3.ExprRef:
    decls = {name: z3.Const(name, sort, ctx=ctx)}
    assertions = z3.parse_smt2_string(
        f"(assert (= {name} {value_str}))", decls=decls, ctx=ctx
    )
    if not assertions:
        raise ValueError(f"Could not parse value for {name}: {value_str}")
    eq = assertions[0]
    return eq.arg(1)


def _check_candidate_external_ipc(
    task: tuple,
) -> tuple:
    script, var_names, solver_name = task
    bin_cmd = _get_solver_cmd(solver_name)
    bin_solver = SMTLIBSolver(bin_cmd)
    status = bin_solver.check_sat_from_scratch(script)
    values = ""
    if status == SolverResult.SAT:
        if var_names:
            values = bin_solver.get_expr_values(var_names)
    bin_solver.stop()
    return status.name, values


def _parallel_check_candidates_ipc(
    fmls: List[z3.BoolRef],
    var_names: List[str],
    logic: str,
    solver_name: str,
    num_workers: int,
) -> List[tuple]:
    tasks = []
    for fml in fmls:
        script = _build_smt2_script(logic, fml)
        tasks.append((script, var_names, solver_name))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_candidate_external_ipc, task) for task in tasks]
        return [f.result() for f in futures]


class ExistsSolver:
    """Exists solver for LIA/LRA/LIRA."""

    def __init__(self, cared_vars: List[z3.ExprRef], phi: z3.BoolRef, logic: str):
        self.x_vars = cared_vars
        self.fmls = [phi]
        self.logic = logic

    def add_constraint(self, fml: z3.BoolRef) -> None:
        self.fmls.append(fml)

    def get_models(
        self,
        num_samples: int,
        strategy: ESolverSampleStrategy = ESolverSampleStrategy.BLOCKING,
        config: Optional[dict] = None,
    ) -> List[z3.ModelRef]:
        try:
            return sample_models(
                self.fmls,
                self.x_vars,
                self.logic,
                num_samples,
                strategy=strategy,
                config=config,
            )
        except SamplingUnknown as exc:
            raise ExistsSolverUnknown() from exc


def _check_candidate(fml: z3.BoolRef, logic: str) -> z3.ModelRef:
    solver = _make_solver(logic, ctx=fml.ctx)
    solver.add(fml)
    res = solver.check()
    if res == z3.sat:
        return solver.model()
    if res == z3.unsat:
        raise ForAllSolverSuccess()
    raise ForAllSolverUnknown()


def _parallel_check_candidates(
    fmls: List[z3.BoolRef], num_workers: int, logic: str
) -> List[z3.ModelRef]:
    tasks = []
    for fml in fmls:
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        tasks.append(i_fml)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_candidate, task, logic) for task in tasks]
        return [f.result() for f in futures]


class ForAllSolver:
    """Forall solver for LIA/LRA/LIRA."""

    def __init__(
        self,
        ctx: z3.Context,
        logic: str,
        solver_name: str = g_forall_bin_solver,
        num_workers: int = g_forall_num_workers,
    ):
        self.ctx = ctx
        self.logic = logic
        self.solver_name = solver_name
        self.num_workers = num_workers
        self._worker_solvers: List[z3.Solver] = []
        self._worker_contexts: List[z3.Context] = []

    def check(self, cnt_list: List[z3.BoolRef]) -> List[z3.ModelRef]:
        if m_forall_solver_strategy == FSolverMode.SEQUENTIAL:
            return self._sequential_check(cnt_list)
        if m_forall_solver_strategy == FSolverMode.PARALLEL_THREAD:
            return self._parallel_check_thread(cnt_list)
        if m_forall_solver_strategy == FSolverMode.PARALLEL_PROCESS_IPC:
            return self._parallel_check_ipc(cnt_list)
        raise NotImplementedError()

    def _sequential_check(self, cnt_list: List[z3.BoolRef]) -> List[z3.ModelRef]:
        models = []
        solver = _make_solver(self.logic, ctx=self.ctx)
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

    def _ensure_worker_pool(self) -> None:
        if self._worker_solvers:
            return
        for _ in range(self.num_workers):
            worker_ctx = z3.Context()
            self._worker_contexts.append(worker_ctx)
            self._worker_solvers.append(_make_solver(self.logic, ctx=worker_ctx))

    def _check_in_worker(self, worker_idx: int, fml: z3.BoolRef) -> z3.ModelRef:
        solver = self._worker_solvers[worker_idx]
        worker_ctx = self._worker_contexts[worker_idx]
        local_fml = fml.translate(worker_ctx)
        solver.push()
        try:
            solver.add(local_fml)
            res = solver.check()
            if res == z3.sat:
                return solver.model()
            if res == z3.unsat:
                raise ForAllSolverSuccess()
            raise ForAllSolverUnknown()
        finally:
            solver.pop()

    def _parallel_check_thread(self, cnt_list: List[z3.BoolRef]) -> List[z3.ModelRef]:
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

    def _parallel_check_ipc(self, cnt_list: List[z3.BoolRef]) -> List[dict]:
        if not cnt_list:
            return []
        var_names = sorted({v.sexpr() for v in get_vars(z3.And(cnt_list))})
        results = _parallel_check_candidates_ipc(
            cnt_list,
            var_names=var_names,
            logic=self.logic,
            solver_name=self.solver_name,
            num_workers=self.num_workers,
        )
        models = []
        var_map = {v.sexpr(): v for v in get_vars(z3.And(cnt_list))}
        for status_name, raw_values in results:
            if status_name == SolverResult.UNSAT.name:
                raise ForAllSolverSuccess()
            if status_name == SolverResult.UNKNOWN.name:
                raise ForAllSolverUnknown()
            values_map = _parse_values_output(raw_values)
            model = {}
            for name, val_str in values_map.items():
                var = var_map.get(name)
                if var is None:
                    continue
                model[var] = _value_from_smt2_string(
                    name, val_str, var.sort(), ctx=self.ctx
                )
            models.append(model)
        return models


def lira_efsmt_with_parallel_cegis(
    exists_vars: List[z3.ExprRef],
    forall_vars: List[z3.ExprRef],
    phi: z3.ExprRef,
    maxloops: Optional[int] = None,
    num_samples: int = 5,
    forall_mode: Optional[FSolverMode] = None,
    bin_solver_name: str = g_forall_bin_solver,
    num_workers: int = g_forall_num_workers,
    sample_strategy: ESolverSampleStrategy = ESolverSampleStrategy.BLOCKING,
    sample_max_tries: int = 25,
    sample_seed_low: int = 1,
    sample_seed_high: int = 1000,
    sample_config: Optional[dict] = None,
) -> EFLIRAResult:
    """Solve exists x. forall y. phi(x, y) with parallel CEGIS."""
    if forall_mode is not None:
        global m_forall_solver_strategy
        m_forall_solver_strategy = forall_mode
    logic = _infer_qf_logic(exists_vars + forall_vars)
    ctx = phi.ctx
    esolver = ExistsSolver(exists_vars, z3.BoolVal(True, ctx=ctx), logic=logic)
    fsolver = ForAllSolver(
        ctx, logic=logic, solver_name=bin_solver_name, num_workers=num_workers
    )
    iterations = 0
    result = EFLIRAResult.UNKNOWN

    try:
        while maxloops is None or iterations <= maxloops:
            logger.debug("Iteration: %s", iterations)
            iterations += 1
            effective_config = {
                "max_tries": sample_max_tries,
                "seed_low": sample_seed_low,
                "seed_high": sample_seed_high,
            }
            if sample_config:
                effective_config.update(sample_config)
            e_models = esolver.get_models(
                num_samples,
                strategy=sample_strategy,
                config=effective_config,
            )
            if len(e_models) == 0:
                result = EFLIRAResult.UNSAT
                break
            reverse_sub_phis = []
            for emodel in e_models:
                x_mappings = [
                    (x, emodel.eval(x, model_completion=True)) for x in exists_vars
                ]
                sub_phi = z3.simplify(z3.substitute(phi, x_mappings))
                reverse_sub_phis.append(z3.Not(sub_phi))

            fmodels = fsolver.check(reverse_sub_phis)
            if len(fmodels) == 0:
                result = EFLIRAResult.SAT
                break
            for fmodel in fmodels:
                if isinstance(fmodel, z3.ModelRef):
                    y_mappings = [
                        (y, fmodel.eval(y, model_completion=True))
                        for y in forall_vars
                    ]
                else:
                    y_mappings = [(y, fmodel.get(y, y)) for y in forall_vars]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                if z3.is_false(sub_phi):
                    raise ExistsSolverSuccess()
                esolver.add_constraint(sub_phi)

    except ForAllSolverSuccess:
        result = EFLIRAResult.SAT
    except ForAllSolverUnknown:
        result = EFLIRAResult.UNKNOWN
    except ExistsSolverSuccess:
        result = EFLIRAResult.UNSAT
    except ExistsSolverUnknown:
        result = EFLIRAResult.UNKNOWN
    except Exception as ex:
        logger.error("Unexpected error: %s", ex)
        result = EFLIRAResult.UNKNOWN

    return result


class ParallelEFLIRASolver(EFLIRASolver):
    """Parallel EFLIRA solver using CEGIS with parallel checks."""

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "cegis")
        self.forall_mode = kwargs.get("forall_mode", None)
        self.bin_solver_name = kwargs.get("bin_solver_name", g_forall_bin_solver)
        self.num_workers = kwargs.get("num_workers", g_forall_num_workers)
        self.sample_strategy = kwargs.get(
            "sample_strategy", ESolverSampleStrategy.BLOCKING
        )
        self.sample_max_tries = kwargs.get("sample_max_tries", 25)
        self.sample_seed_low = kwargs.get("sample_seed_low", 1)
        self.sample_seed_high = kwargs.get("sample_seed_high", 1000)
        self.sample_config = kwargs.get("sample_config", None)

    def solve_efsmt_lira(
        self, existential_vars: List[z3.ExprRef], universal_vars: List[z3.ExprRef], phi
    ) -> EFLIRAResult:
        if self.mode == "cegis":
            return lira_efsmt_with_parallel_cegis(
                existential_vars,
                universal_vars,
                phi,
                forall_mode=self.forall_mode,
                bin_solver_name=self.bin_solver_name,
                num_workers=self.num_workers,
                sample_strategy=self.sample_strategy,
                sample_max_tries=self.sample_max_tries,
                sample_seed_low=self.sample_seed_low,
                sample_seed_high=self.sample_seed_high,
                sample_config=self.sample_config,
            )
        raise NotImplementedError()


def test_efsmt():
    """Test function for EFLIRA solver."""
    x, y = z3.Ints("x y")
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    solver = ParallelEFLIRASolver(mode="cegis")
    result = solver.solve_efsmt_lira([x], [y], fmla)
    print(f"Result: {result}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_efsmt()
