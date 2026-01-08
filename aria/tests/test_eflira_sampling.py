import random

import z3

from aria.quant.eflira.eflira_sampling_utils import (
    ESolverSampleStrategy,
    sample_models,
)
from aria.quant.eflira.eflira_parallel import (
    EFLIRAResult,
    lira_efsmt_with_parallel_cegis,
)


def _models_satisfy(models, fmls):
    for model in models:
        for fml in fmls:
            if not z3.is_true(model.eval(fml, model_completion=True)):
                return False
    return True


def test_eflira_sampling_strategies_return_models():
    random.seed(0)
    x = z3.Int("x")
    fmls = [x >= 0, x <= 10]
    logic = "QF_LIA"

    strategies = [
        ESolverSampleStrategy.BLOCKING,
        ESolverSampleStrategy.RANDOM_SEED,
        ESolverSampleStrategy.OPTIMIZE,
        ESolverSampleStrategy.LEXICOGRAPHIC,
        ESolverSampleStrategy.JITTER,
        ESolverSampleStrategy.PORTFOLIO,
    ]

    for strategy in strategies:
        models = sample_models(
            fmls,
            [x],
            logic,
            num_samples=3,
            strategy=strategy,
            config={
                "max_tries": 20,
                "optimize_max_tries": 10,
                "optimize_objectives": 2,
                "jitter_int_delta": 1,
                "jitter_max_tries": 20,
            },
        )
        assert len(models) >= 1
        assert _models_satisfy(models, fmls)


def test_eflira_cegis_sat_with_portfolio_sampling():
    x, y = z3.Ints("x y")
    phi = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    res = lira_efsmt_with_parallel_cegis(
        [x],
        [y],
        phi,
        maxloops=20,
        num_samples=2,
        sample_strategy=ESolverSampleStrategy.PORTFOLIO,
    )
    assert res == EFLIRAResult.SAT


def test_eflira_cegis_unsat_with_blocking_sampling():
    x, y = z3.Ints("x y")
    phi = z3.BoolVal(False)
    res = lira_efsmt_with_parallel_cegis(
        [x],
        [y],
        phi,
        maxloops=5,
        num_samples=1,
        sample_strategy=ESolverSampleStrategy.BLOCKING,
    )
    assert res == EFLIRAResult.UNSAT
