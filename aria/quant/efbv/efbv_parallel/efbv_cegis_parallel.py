"""Solving Exists-Forall Problem (currently focus on bit-vec?).

https://github.com/pysmt/pysmt/blob/97088bf3b0d64137c3099ef79a4e153b10ccfda7/examples/efsmt.py

Possible extensions:
- better generalization for esolver
- better generalization for fsolver
- uniform sampling for processing multiple models each round?
- use unsat core??

However, the counterexample may not be general enough to exclude a large
class of invalid expressions, which will lead to the repetition of several
loop iterations. We believe our sampling technique could be a good
enhancement to CEGIS. By generating several diverse counterexamples, the
verifier can provide more information to the learner so that it can make
more progress on its own, limiting the number of calls to the verifier
"""

import logging
import time
from typing import List

import z3

from aria.quant.efbv.efbv_parallel.efbv_exists_solver import ExistsSolver
from aria.quant.efbv.efbv_parallel.efbv_forall_solver import ForAllSolver
from aria.quant.efbv.efbv_parallel.efbv_utils import EFBVResult, EFBVTactic, EFBVSolver
from aria.quant.efbv.efbv_parallel.exceptions import (
    ExitsSolverSuccess,
    ExitsSolverUnknown,
    ForAllSolverSuccess,
    ForAllSolverUnknown,
)

logger = logging.getLogger(__name__)

g_efbv_tactic = EFBVTactic.Z3_QBF


def bv_efsmt_with_uniform_sampling(
    exists_vars, forall_vars, phi, maxloops=None, num_samples: int = 5
):
    """
    Solve exists x. forall y. phi(x, y) using uniform sampling.

    Args:
        exists_vars: List of existential variables
        forall_vars: List of universal variables
        phi: Formula to solve
        maxloops: Maximum number of iterations (None for unlimited)
        num_samples: Number of samples to generate per iteration

    Returns:
        EFBVResult indicating SAT/UNSAT/UNKNOWN
    """
    # x = [item for item in get_vars(phi) if item not in y]

    esolver = ExistsSolver(exists_vars, z3.BoolVal(True))
    fsolver = ForAllSolver(exists_vars[0].ctx)
    # fsolver.vars = forall_vars
    # fsolver.phi = phi

    iterations = 0
    result = EFBVResult.UNKNOWN
    try:
        while maxloops is None or iterations <= maxloops:
            logger.debug(f"Iteration: {iterations}")
            iterations += 1
            # TODO: need to make the fist and the subsequent iteration different???
            # TODO: in the uniform sampler, I always call the solver once before xx...
            # Get multiple exist models
            e_models = esolver.get_models(num_samples)

            if len(e_models) == 0:
                logger.debug("  Success with UNSAT")
                result = EFBVResult.UNSAT  # esolver tells unsat
                break
            # sub_phis = []
            reverse_sub_phis = []
            # print("e models: ", e_models)
            for emodel in e_models:
                x_mappings = [
                    (x, emodel.eval(x, model_completion=True)) for x in exists_vars
                ]
                sub_phi = z3.simplify(z3.substitute(phi, x_mappings))
                # sub_phis.append(sub_phi)
                reverse_sub_phis.append(z3.Not(sub_phi))

            fmodels = fsolver.check(reverse_sub_phis)
            if len(fmodels) == 0:
                logger.debug("  Success with SAT")
                result = EFBVResult.SAT  # fsolver tells sat
                break
            # Add new constraints from forall models
            for fmodel in fmodels:
                # sigma = [model.eval(vy, True) for vy in self.forall_vars]
                y_mappings = [
                    (y, fmodel.eval(y, model_completion=True)) for y in forall_vars
                ]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                # block all CEX?
                # print("blocking fml: ", sub_phi)
                if z3.is_false(sub_phi):
                    logger.debug("  Success with UNSAT")
                    # using break is not fine
                    raise ExitsSolverSuccess()
                esolver.fmls.append(sub_phi)

    except ForAllSolverSuccess:
        logger.debug("Forall solver success - SAT")
        result = EFBVResult.SAT
    except ForAllSolverUnknown:
        logger.debug("  Forall solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except ExitsSolverSuccess:
        logger.debug("Exists solver success - UNSAT")
        result = EFBVResult.UNSAT
    except ExitsSolverUnknown:
        logger.debug("  Exists solver UNKNOWN")
        result = EFBVResult.UNKNOWN
    except Exception as ex:
        logger.error("Unexpected error: %s", ex)
        result = EFBVResult.UNKNOWN

    return result


class ParallelEFBVSolver(EFBVSolver):
    """Parallel EFBV solver using CEGIS with uniform sampling."""

    def __init__(self, **kwargs):
        """Initialize parallel EFBV solver.

        Args:
            mode: Solver mode (e.g., "canary", "qbf", "simple_cegar", "z3")
        """
        super().__init__(**kwargs)
        self.mode = kwargs.get("mode", "canary")

    def solve_efsmt_bv(
        self, existential_vars: List, universal_vars: List, phi: z3.ExprRef
    ):
        """Solve EFBV problem."""
        if self.mode == "canary":
            return bv_efsmt_with_uniform_sampling(existential_vars, universal_vars, phi)
        raise NotImplementedError()


def test_efsmt():
    """Test function for EFBV solver."""
    x, y, z = z3.BitVecs("x y z", 16)
    fmla = z3.Implies(z3.And(y > 0, y < 10), y - 2 * x < 7)
    #
    start = time.time()
    solver = ParallelEFBVSolver(mode="canary")
    result = solver.solve_efsmt_bv([x], [y], fmla)
    duration = time.time() - start
    print(f"Result: {result}")
    print(f"Time: {duration:.3f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_efsmt()
