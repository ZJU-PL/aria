"""Sampling utilities for EFLIRA exists models."""

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Dict, List, Optional

import z3

logger = logging.getLogger(__name__)


class SamplingUnknown(Exception):
    """Sampling returned UNKNOWN."""


class ESolverSampleStrategy(Enum):
    BLOCKING = 0
    RANDOM_SEED = 1
    OPTIMIZE = 2
    LEXICOGRAPHIC = 3
    JITTER = 4
    PORTFOLIO = 5


DEFAULT_SAMPLING_CONFIG: Dict = {
    "max_tries": 25,
    "seed_low": 1,
    "seed_high": 1000,
    "optimize_objectives": 4,
    "optimize_max_tries": 20,
    "optimize_coeff_low": 1,
    "optimize_coeff_high": 5,
    "lex_order": None,
    "jitter_int_delta": 1,
    "jitter_real_delta": "0.1",
    "jitter_max_tries": 25,
    "portfolio": [
        ESolverSampleStrategy.BLOCKING,
        ESolverSampleStrategy.RANDOM_SEED,
        ESolverSampleStrategy.OPTIMIZE,
        ESolverSampleStrategy.JITTER,
        ESolverSampleStrategy.LEXICOGRAPHIC,
    ],
}


def _merge_config(config: Optional[Dict]) -> Dict:
    merged = dict(DEFAULT_SAMPLING_CONFIG)
    if config:
        merged.update(config)
    return merged


def _make_solver(logic: str, ctx: Optional[z3.Context] = None) -> z3.Solver:
    try:
        return z3.SolverFor(logic, ctx=ctx)
    except z3.Z3Exception:
        return z3.Solver(ctx=ctx)


def _model_key(x_vars: List[z3.ExprRef], model: z3.ModelRef) -> tuple:
    key = []
    for x in x_vars:
        val = model.eval(x, model_completion=True)
        key.append((x.sexpr(), val.sexpr()))
    return tuple(key)


def _block_model(x_vars: List[z3.ExprRef], model: z3.ModelRef) -> z3.BoolRef:
    if not x_vars:
        return z3.BoolVal(False)
    diffs = [x != model.eval(x, model_completion=True) for x in x_vars]
    return z3.Or(diffs)


def _lex_gt(x_vars: List[z3.ExprRef], model: z3.ModelRef) -> z3.BoolRef:
    if not x_vars:
        return z3.BoolVal(False)
    prefix = []
    clauses = []
    for x in x_vars:
        val = model.eval(x, model_completion=True)
        clauses.append(z3.And(prefix + [x > val]))
        prefix.append(x == val)
    return z3.Or(clauses)


def _get_models_blocking(
    fmls: List[z3.BoolRef], x_vars: List[z3.ExprRef], logic: str, num_samples: int
) -> List[z3.ModelRef]:
    sol = _make_solver(logic)
    sol.add(z3.And(fmls))
    res = sol.check()
    if res == z3.unsat:
        return []
    if res == z3.unknown:
        raise SamplingUnknown()
    models = [sol.model()]
    if num_samples <= 1:
        return models
    for _ in range(num_samples - 1):
        sol.add(_block_model(x_vars, models[-1]))
        res = sol.check()
        if res == z3.sat:
            models.append(sol.model())
        elif res == z3.unsat:
            break
        else:
            raise SamplingUnknown()
    return models


def _get_models_random_seed(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    max_tries: int,
    seed_low: int,
    seed_high: int,
) -> List[z3.ModelRef]:
    models = []
    seen = set()
    tries = 0
    while len(models) < num_samples and tries < max_tries:
        tries += 1
        sol = _make_solver(logic)
        sol.add(z3.And(fmls))
        sol.set("random_seed", random.randint(seed_low, seed_high))
        res = sol.check()
        if res == z3.unsat:
            break
        if res == z3.unknown:
            raise SamplingUnknown()
        model = sol.model()
        key = _model_key(x_vars, model)
        if key in seen:
            continue
        seen.add(key)
        models.append(model)
    return models


def _random_linear_objective(
    x_vars: List[z3.ExprRef], coeff_low: int, coeff_high: int
) -> Optional[z3.ArithRef]:
    terms = []
    for x in x_vars:
        coeff = random.randint(coeff_low, coeff_high)
        if random.choice([True, False]):
            coeff = -coeff
        terms.append(coeff * x)
    if not terms:
        return None
    obj = terms[0]
    for term in terms[1:]:
        obj = obj + term
    return obj


def _get_models_optimize(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    max_tries: int,
    objectives: int,
    coeff_low: int,
    coeff_high: int,
) -> List[z3.ModelRef]:
    models = []
    seen = set()
    tries = 0
    while len(models) < num_samples and tries < max_tries:
        tries += 1
        opt = z3.Optimize()
        opt.add(z3.And(fmls))
        for _ in range(objectives):
            obj = _random_linear_objective(x_vars, coeff_low, coeff_high)
            if obj is None:
                break
            if random.choice([True, False]):
                opt.maximize(obj)
            else:
                opt.minimize(obj)
        res = opt.check()
        if res == z3.unsat:
            break
        if res == z3.unknown:
            raise SamplingUnknown()
        model = opt.model()
        key = _model_key(x_vars, model)
        if key in seen:
            continue
        seen.add(key)
        models.append(model)
    return models


def _order_vars(
    x_vars: List[z3.ExprRef], lex_order: Optional[List[str]]
) -> List[z3.ExprRef]:
    if not lex_order:
        return list(x_vars)
    by_name = {v.sexpr(): v for v in x_vars}
    ordered = []
    for name in lex_order:
        var = by_name.get(name)
        if var is not None:
            ordered.append(var)
    for v in x_vars:
        if v not in ordered:
            ordered.append(v)
    return ordered


def _get_models_lexicographic(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    lex_order: Optional[List[str]],
) -> List[z3.ModelRef]:
    ordered_vars = _order_vars(x_vars, lex_order)
    sol = _make_solver(logic)
    sol.add(z3.And(fmls))
    res = sol.check()
    if res == z3.unsat:
        return []
    if res == z3.unknown:
        raise SamplingUnknown()
    models = [sol.model()]
    if num_samples <= 1:
        return models
    for _ in range(num_samples - 1):
        sol.add(_lex_gt(ordered_vars, models[-1]))
        res = sol.check()
        if res == z3.sat:
            models.append(sol.model())
        elif res == z3.unsat:
            break
        else:
            raise SamplingUnknown()
    return models


def _get_models_jitter(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    int_delta: int,
    real_delta: str,
    max_tries: int,
) -> List[z3.ModelRef]:
    models = []
    seen = set()
    base_models = _get_models_blocking(fmls, x_vars, logic, 1)
    if not base_models:
        return []
    models.extend(base_models)
    for m in base_models:
        seen.add(_model_key(x_vars, m))
    tries = 0
    while len(models) < num_samples and tries < max_tries:
        tries += 1
        base = random.choice(models)
        sol = _make_solver(logic)
        sol.add(z3.And(fmls))
        jitter_clauses = []
        for x in x_vars:
            val = base.eval(x, model_completion=True)
            if x.sort().kind() == z3.Z3_INT_SORT:
                if int_delta <= 0:
                    continue
                delta = z3.IntVal(int_delta)
            else:
                delta = z3.RealVal(real_delta)
            if random.choice([True, False]):
                jitter_clauses.append(x >= val + delta)
            else:
                jitter_clauses.append(x <= val - delta)
        if jitter_clauses:
            sol.add(z3.Or(jitter_clauses))
        res = sol.check()
        if res == z3.unsat:
            continue
        if res == z3.unknown:
            raise SamplingUnknown()
        model = sol.model()
        key = _model_key(x_vars, model)
        if key in seen:
            continue
        seen.add(key)
        models.append(model)
    return models


def _get_models_portfolio(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    config: Dict,
) -> List[z3.ModelRef]:
    models = []
    seen = set()
    strategies = config.get("portfolio", DEFAULT_SAMPLING_CONFIG["portfolio"])
    for strategy in strategies:
        if len(models) >= num_samples:
            break
        if strategy == ESolverSampleStrategy.PORTFOLIO:
            continue
        batch = sample_models(
            fmls,
            x_vars,
            logic,
            num_samples - len(models),
            strategy=strategy,
            config=config,
        )
        for m in batch:
            key = _model_key(x_vars, m)
            if key in seen:
                continue
            seen.add(key)
            models.append(m)
    return models


def sample_models(
    fmls: List[z3.BoolRef],
    x_vars: List[z3.ExprRef],
    logic: str,
    num_samples: int,
    strategy: ESolverSampleStrategy,
    config: Optional[Dict] = None,
) -> List[z3.ModelRef]:
    cfg = _merge_config(config)
    if strategy == ESolverSampleStrategy.BLOCKING:
        return _get_models_blocking(fmls, x_vars, logic, num_samples)
    if strategy == ESolverSampleStrategy.RANDOM_SEED:
        return _get_models_random_seed(
            fmls,
            x_vars,
            logic,
            num_samples,
            max_tries=cfg["max_tries"],
            seed_low=cfg["seed_low"],
            seed_high=cfg["seed_high"],
        )
    if strategy == ESolverSampleStrategy.OPTIMIZE:
        return _get_models_optimize(
            fmls,
            x_vars,
            logic,
            num_samples,
            max_tries=cfg["optimize_max_tries"],
            objectives=cfg["optimize_objectives"],
            coeff_low=cfg["optimize_coeff_low"],
            coeff_high=cfg["optimize_coeff_high"],
        )
    if strategy == ESolverSampleStrategy.LEXICOGRAPHIC:
        return _get_models_lexicographic(
            fmls, x_vars, logic, num_samples, lex_order=cfg["lex_order"]
        )
    if strategy == ESolverSampleStrategy.JITTER:
        return _get_models_jitter(
            fmls,
            x_vars,
            logic,
            num_samples,
            int_delta=cfg["jitter_int_delta"],
            real_delta=cfg["jitter_real_delta"],
            max_tries=cfg["jitter_max_tries"],
        )
    if strategy == ESolverSampleStrategy.PORTFOLIO:
        return _get_models_portfolio(fmls, x_vars, logic, num_samples, cfg)
    raise NotImplementedError()
