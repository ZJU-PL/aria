"""
This module provides a BitVector model counter for Z3 formulas.
It includes functions for counting models
using enumeration, parallel enumeration, and sharpSAT.
"""

from typing import List
import itertools
import logging
from timeit import default_timer as counting_timer

import z3
from aria.utils.z3_expr_utils import get_variables
from aria.smt.bv.mapped_blast import translate_smt2formula_to_cnf
from aria.counting.bool.dimacs_counting import count_dimacs_solutions_parallel


def split_list(alist, wanted_parts=1):
    """
    Split a list into wanted_parts number of parts.

    Args:
        alist (list): The list to be split.
        wanted_parts (int): The number of parts to split the list into.

    Returns:
        list: A list of lists, where each sublist is a part of the original list.

    Raises:
        ZeroDivisionError: If wanted_parts is zero.
    """
    if wanted_parts == 0:
        raise ZeroDivisionError("wanted_parts must be greater than zero.")
    length = len(alist)
    return [
        alist[i * length // wanted_parts : (i + 1) * length // wanted_parts]
        for i in range(wanted_parts)
    ]


def check_candidate_model(formula, all_vars, candidate):
    """Return True iff candidate assignment satisfies formula."""
    solver = z3.Solver(ctx=formula.ctx)
    solver.add(formula)
    assumptions = [
        var == z3.BitVecVal(value, var.sort().size())
        for var, value in zip(all_vars, candidate)
    ]
    return solver.check(assumptions) == z3.sat


def check_candidate_models_set(formula: z3.ExprRef, assignments: List) -> int:
    """Count satisfying assignments in the given assignment set."""
    variables = get_variables(formula)
    num_solutions = sum(
        1 for cand in assignments if check_candidate_model(formula, variables, cand)
    )
    logging.info("num solutions in subset: %d", num_solutions)
    return num_solutions


class BVModelCounter:
    """
        A class for counting the number of models of a Z3 formula.
    Attributes:
        formula (z3.ExprRef): The Z3 formula to count models for.
        vars (list): A list of all variables in the formula.
    """

    def __init__(self):
        self.formula = None
        self.vars = []
        self.counts = 0
        self.smt2file = None

    def init_from_file(self, filename):
        """
        Initialize the model counter from an SMT2 file.

        Args:
            filename: Path to the SMT2 file

        Returns:
            None: Always returns None (raises exception on error)
        """
        try:
            self.smt2file = filename
            self.formula = z3.And(z3.parse_smt2_file(filename))
            for var in get_variables(self.formula):
                if z3.is_bv(var):
                    self.vars.append(var)
            logging.debug("Init model counter success!")
        except z3.Z3Exception as ex:
            logging.error(ex)
            raise

    def init_from_fml(self, fml):
        """
        Initialize the model counter from a Z3 formula.

        Args:
            fml: Z3 formula to count models for

        Returns:
            None: Always returns None (raises exception on error)
        """
        try:
            self.formula = fml
            for var in get_variables(self.formula):
                if z3.is_bv(var):
                    self.vars.append(var)
        except z3.Z3Exception as ex:
            logging.error(ex)
            raise

    def count_model_by_bv_enumeration(self):
        """
        Enumerate all possible assignments
        TODO: handle signed and the mixing of signed and unsigned
        """
        # time_start = time.process_time()
        time_start = counting_timer()
        logging.debug("Start BV enumeration-based")
        domains = [tuple(range(0, 2 ** v.sort().size())) for v in self.vars]
        solutions = sum(
            1
            for assignment in itertools.product(*domains)
            if check_candidate_model(self.formula, self.vars, assignment)
        )
        logging.info("Time: %s", counting_timer() - time_start)
        logging.info("BV enumeration total solutions: %d", solutions)
        return solutions, counting_timer() - time_start

    def count_model_by_enumeration_parallel(self):
        """Parallel enumeration is not implemented."""
        raise NotImplementedError("Parallel BV enumeration is not implemented.")

    def count_models_by_sharp_sat(self):
        """
        Count models using sharpSAT solver.

        Returns:
            tuple: (solutions, time_taken) where solutions is the count and time_taken is elapsed time
        """
        _, _, header, clauses = translate_smt2formula_to_cnf(self.formula)
        time_start = counting_timer()
        # solutions = count_dimacs_solutions(header, clauses)
        solutions = count_dimacs_solutions_parallel(header, clauses)
        logging.info("Time: %s", counting_timer() - time_start)
        logging.info("sharpSAT total solutions: %d", solutions)
        return solutions, counting_timer() - time_start


def feat_test():
    """Test function for model counting features."""
    mc = BVModelCounter()
    x = z3.BitVec("x", 4)
    y = z3.BitVec("y", 4)
    fml = z3.And(z3.UGT(x, 0), z3.UGT(y, 0), z3.ULT(x - y, 10))
    mc.init_from_fml(fml)
    # mc.init_from_file('../../benchmarks/t1.smt2')
    # mc.count_model_by_bv_enumeration()
    mc.count_models_by_sharp_sat()


if __name__ == "__main__":
    feat_test()
