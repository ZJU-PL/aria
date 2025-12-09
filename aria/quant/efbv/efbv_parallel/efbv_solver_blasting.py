"""
In this engine, we bit-blast the BV formula to Boolean formulas, based on whcih
we run (parallel) uniform sampling.

However, the implementation is a bit complicated.
E.g., we need to perform bit-blasting again and again
"""
# FIXME: this file is not used for now (a bit complicated communication...)

import logging
import time
from typing import List, Tuple

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from aria.utils.z3_expr_utils import get_variables
from aria.smt.bv import translate_smt2formula_to_cnf
from aria.utils import SolverResult
from aria.quant.efbv.efbv_parallel.exceptions import ForAllSolverSuccess

from aria.quant.efbv.efbv_parallel.efbv_utils import EFBVResult

logger = logging.getLogger(__name__)


# maintain a "global pool"?
# e_solver_constraints = []


class SeqExistsSolver:
    """
    TODO: it seems we need to create multiple objects
    """

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

    def sample_boolean_model(self):
        """Check satisfiability of a bit-vector formula
        If it is satisfiable, sample a set of models"""
        clauses_numeric = self.bit_blast()
        # Main difficulty: how to infer signedness of each variable
        cnf = CNF(from_clauses=clauses_numeric)
        name = "minisat22"
        try:
            start_time = time.time()
            with Solver(name=name, bootstrap_with=cnf) as solver:
                if not solver.solve():
                    return SolverResult.UNSAT, []
                # TODO: figure out what is the order of the vars in the boolean model
                bool_model = solver.get_model()
                logger.debug("SAT solving time: {}".format(time.time() - start_time))
                return SolverResult.SAT, bool_model

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


class SeqForAllSolver:
    """ForAllSolver
    """

    def __init__(self):
        self.phi = None
        self.bv_size = 16
        self.universal_vars = []

    def from_smt_formula(self, formula: z3.BoolRef, uvars: List[z3.ExprRef]):
        self.phi = formula
        self.universal_vars = uvars

    def check_model(self, raw_mappings: List[Tuple[str, int]]):
        """Validate models produced by the exists solver
        :param raw_mappings: [("x1", 3), ("x2", 10), ..., ()]
        :return: counter-example? (in the form of another mapping)
        """
        # e.g., mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
        z3_mappings = []
        for var_name, var_value in raw_mappings:
            z3_mappings.append((z3.BitVec(var_name, self.bv_size), var_value))  # TODO: need q quick test
            # z3_mappings.append((z3.BitVec(var_name, self.bv_size), var_value))  # do we need this?

        sub_phi = z3.simplify(z3.substitute(self.phi, z3_mappings))
        sol = z3.Solver()
        sol.add(z3.Not(sub_phi))  # z3.Not(sub_phi) is the condition to be checked/solved
        if sol.check() == z3.unsat:
            raise ForAllSolverSuccess()  # a "dirty" trick for exist (TODO: fix this?)
        else:
            fmodel = sol.model()
            y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in self.universal_vars]
            return y_mappings

    def validate_models(self, all_mappings):
        res = []
        for mappings in all_mappings:
            res.append(self.validate_model(mappings))
        return res


class SeqEFBVSolver:
    def __init__(self):
        self.fml = None
        self.bv2bool = {}  # map a bit-vector variable to a list of Boolean variables [ordered by bit?]
        self.bool2id = {}  # map a Boolean variable to its internal ID in pysat
        self.vars = []
        self.verbose = 0
        self.signed = False

    def from_smt_formula(self, formula: z3.BoolRef):
        self.fml = formula
        self.vars = get_variables(self.fml)

    def solve_internal(self, models):
        """Check Boolean models produced by the "exists solver"
        1. Sample Boolean models
        2. Build bit-vector models M = {M1,...,Mn} from the Boolean models
        3. Build bit-vector formulas F1,..., Fn for the "forall solver"
        4. If any Fi is unsatisfiable, break and return "SAT"
        5. If some Fi is satisfiable, build a blocking formula Gi (from a model of Fi)

        FIXME: in principle, in 4, we can also use multiple Boolean models.
        :return:
        """
        return

    def solve(self, exists_vars: List[z3.ExprRef], forall_vars: List[z3.ExprRef], phi: z3.ExprRef,
              maxloops=None) -> EFBVResult:
        """
        Solves exists x. forall y. phi(x, y)
        FIXME: inconsistent with efsmt
        """
        return EFBVResult.UNKNOWN
