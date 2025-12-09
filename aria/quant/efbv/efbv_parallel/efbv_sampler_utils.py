"""
Using (parallel) Boolean model samplers to sample bit-vector models
- Track the correlations between Boolean and Bit-vector level information
- Run external samplers and build bit-vector models from the Boolean models
"""

import logging
import concurrent.futures
from random import randrange
from typing import List, Tuple

import z3

from aria.smt.bv import translate_smt2formula_to_cnf

logger = logging.getLogger(__name__)


class BitBlastSampler:
    def __init__(self):
        self.fml = None
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat
        self.vars = []
        self.verbose = 0
        self.signed = False

    def bit_blast(self):
        logger.debug("Start translating to CNF...")
        # NOTICE: can be slow
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.fml)
        self.bv2bool = bv2bool
        self.bool2id = id_table
        logger.debug("  from bv to bools: {}".format(self.bv2bool))
        logger.debug("  from bool to pysat id: {}".format(self.bool2id))

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        return clauses_numeric

    def sample_boolean_models(self):
        """Check satisfiability of a bit-vector formula
        If it is satisfiable, sample a set of models"""
        clauses_numeric = self.bit_blast()
        # Main difficulty: how to infer signedness of each variable
        try:
            x = 1
        except Exception as ex:
            print(ex)

    def build_bv_model(self, bool_model) -> List[Tuple[str, int]]:
        """Building `bv models' (used for building candidate bv formulas)"""
        bv_model = {}
        if not self.signed:  # unsigned
            for bv_var in self.bv2bool:
                bool_vars = self.bv2bool[bv_var]
                start = self.bool2id[bool_vars[0]]  # start ID
                bv_val = 0
                for i in range(len(bool_vars)):
                    if bool_model[i + start - 1] > 0:
                        bv_val += 2 ** i
                bv_model[str(bv_var)] = bv_val
        else:  # signed
            # FIXME: the following seems to be wrong
            for bv_var in self.bv2bool:
                bool_vars = self.bv2bool[bv_var]
                start = self.bool2id[bool_vars[0]]  # start ID
                bv_val = 0
                for i in range(len(bool_vars) - 1):
                    if bool_model[i + start - 1] > 0:
                        bv_val += 2 ** i
                if bool_model[len(bool_vars) - 1 + start - 1] > 0:
                    bv_val = -bv_val
                bv_model[str(bv_var)] = bv_val
        # TODO: map back to bit-vector model
        return bv_model


def sample_worker(fml: z3.BoolRef, cared_bits: List):
    """
    :param fml: the formula to be checked
    :param cared_bits: used for sampling (...)
    :return A model
      TODO: allow for sampling more than one models
    """
    # print("Sampling in one thread ...")
    local_ctx = fml.ctx
    solver = z3.SolverFor("QF_BV", ctx=local_ctx)
    solver.add(fml)
    while True:
        rounds = 3  # why 3?
        assumption = z3.BoolVal(True, ctx=local_ctx)
        for _ in range(rounds):
            trials = 10
            fml = z3.BoolVal(randrange(0, 2), ctx=local_ctx)
            for _ in range(trials):
                fml = z3.Xor(fml, cared_bits[randrange(0, len(cared_bits))], ctx=local_ctx)
            assumption = z3.And(assumption, fml)
        if solver.check(assumption) == z3.sat:
            return solver.model()


def parallel_sample(fml: z3.BoolRef, cared_bits: List, num_samples: int, num_workers: int):
    """
    Perform uniform sampling in parallel
    TODO: the cared_bits is only specific the algorithm we use for sampling (which is not good)
      As we may use other algorithms for the sampling
    """
    tasks = []
    # Create new context for the computation
    # Note that we need to do this sequentially, as parallel access to the current context or its objects
    # will result in a segfault
    # origin_ctx = fmls[0].ctx
    for _ in range(num_samples):
        # tasks.append((fml, main_ctx()))
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        i_cared_bits = [bit.translate(i_context) for bit in cared_bits]
        tasks.append((i_fml, i_cared_bits))

    # TODO: try processes?
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #  with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(sample_worker, task[0], task[1]) for task in tasks]
        results = [f.result() for f in futures]
        return results
