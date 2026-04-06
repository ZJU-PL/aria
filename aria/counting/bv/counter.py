"""Bit-vector model counting interfaces."""

import itertools
import logging
from timeit import default_timer as counting_timer
from typing import List, Optional, Sequence, cast

import z3

from aria.counting.bool.dimacs_counting import call_approxmc
from aria.counting.bool.dimacs_counting import count_dimacs_solutions_parallel
from aria.counting.core import CountResult, exact_count_result, unsupported_count_result
from aria.smt.bv.mapped_blast import translate_smt2formula_to_cnf
from aria.utils.z3.expr import get_variables


def split_list(alist, wanted_parts=1):
    """Split a list into wanted_parts number of parts."""
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
        var == z3.BitVecVal(value, _bit_width(var))
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


def _bit_width(var: z3.ExprRef) -> int:
    """Return the bit-width for a bit-vector variable."""

    return cast(z3.BitVecSortRef, var.sort()).size()


class BVModelCounter:
    """Count satisfying assignments for bit-vector formulas."""

    def __init__(self):
        self.formula: Optional[z3.ExprRef] = None
        self.vars = []
        self.counts = 0
        self.smt2file = None

    def init_from_file(self, filename):
        """Initialize the model counter from an SMT2 file."""
        try:
            self.smt2file = filename
            self.vars = []
            formula = cast(z3.ExprRef, z3.And(*z3.parse_smt2_file(filename)))
            self.formula = formula
            for var in get_variables(formula):
                if z3.is_bv(var):
                    self.vars.append(var)
            logging.debug("Init model counter success!")
        except z3.Z3Exception as ex:
            logging.error(ex)
            raise

    def init_from_fml(self, fml):
        """Initialize the model counter from a Z3 formula."""
        try:
            self.vars = []
            formula = cast(z3.ExprRef, fml)
            self.formula = formula
            for var in get_variables(formula):
                if z3.is_bv(var):
                    self.vars.append(var)
        except z3.Z3Exception as ex:
            logging.error(ex)
            raise

    def count_model_by_bv_enumeration_result(
        self, variables: Optional[Sequence[z3.ExprRef]] = None
    ) -> CountResult:
        """Enumerate all bit-vector assignments and return a structured result."""
        if self.formula is None:
            return unsupported_count_result(
                backend="bv-enumeration",
                reason="formula is not initialized",
            )
        projection_vars = list(variables) if variables is not None else self.vars
        available_var_ids = {var.get_id() for var in self.vars}
        for var in projection_vars:
            if var.get_id() not in available_var_ids:
                return unsupported_count_result(
                    backend="bv-enumeration",
                    reason=f"projection variable not found in formula: {var}",
                    projection=[str(item) for item in projection_vars],
                )

        time_start = counting_timer()
        logging.debug("Start BV enumeration-based")
        domains = [tuple(range(0, 2 ** _bit_width(v))) for v in projection_vars]
        solutions = sum(
            1
            for assignment in itertools.product(*domains)
            if check_candidate_model(self.formula, projection_vars, assignment)
        )
        runtime_s = counting_timer() - time_start
        logging.info("Time: %s", runtime_s)
        logging.info("BV enumeration total solutions: %d", solutions)
        return exact_count_result(
            float(solutions),
            backend="bv-enumeration",
            runtime_s=runtime_s,
            projection=[str(var) for var in projection_vars],
            num_variables=len(self.vars),
            projected_variables=len(projection_vars),
            total_bit_width=sum(_bit_width(var) for var in self.vars),
            projected_bit_width=sum(_bit_width(var) for var in projection_vars),
        )

    def count_model_by_enumeration_parallel(self):
        """Parallel enumeration is not implemented."""
        raise NotImplementedError("Parallel BV enumeration is not implemented.")

    def count_models_by_sharp_sat_result(
        self, variables: Optional[Sequence[z3.ExprRef]] = None
    ) -> CountResult:
        """Count models via bit-blasting and sharpSAT."""
        if self.formula is None:
            return unsupported_count_result(
                backend="bv-sharpsat",
                reason="formula is not initialized",
            )
        if variables is not None:
            return unsupported_count_result(
                backend="bv-sharpsat",
                reason="projected sharpSAT counting is not supported",
                projection=[str(var) for var in variables],
            )
        formula = self.formula
        _, _, header, clauses = translate_smt2formula_to_cnf(formula)
        time_start = counting_timer()
        solutions = count_dimacs_solutions_parallel(header, clauses)
        runtime_s = counting_timer() - time_start
        logging.info("Time: %s", runtime_s)
        logging.info("sharpSAT total solutions: %d", solutions)
        return exact_count_result(
            float(solutions),
            backend="bv-sharpsat",
            runtime_s=runtime_s,
            num_variables=len(self.vars),
            total_bit_width=sum(_bit_width(var) for var in self.vars),
        )

    def count_models_by_approxmc_result(
        self, variables: Optional[Sequence[z3.ExprRef]] = None
    ) -> CountResult:
        """Count models approximately via bit-blasting and ApproxMC."""
        if self.formula is None:
            return unsupported_count_result(
                backend="bv-approxmc",
                reason="formula is not initialized",
            )
        if variables is not None:
            return unsupported_count_result(
                backend="bv-approxmc",
                reason="approximate projected BV counting is not supported",
                projection=[str(var) for var in variables],
            )

        time_start = counting_timer()
        _, _, _, clauses = translate_smt2formula_to_cnf(self.formula)
        approx_count = call_approxmc(clauses)
        if approx_count < 0:
            return unsupported_count_result(
                backend="bv-approxmc",
                reason="ApproxMC backend is unavailable or failed",
                runtime_s=counting_timer() - time_start,
            )
        return CountResult(
            status="approximate",
            count=float(approx_count),
            backend="bv-approxmc",
            exact=False,
            runtime_s=counting_timer() - time_start,
            projection=[str(var) for var in self.vars],
            metadata={
                "num_variables": len(self.vars),
                "total_bit_width": sum(_bit_width(var) for var in self.vars),
            },
        )

    def count_models(
        self,
        method: str = "auto",
        variables: Optional[Sequence[z3.ExprRef]] = None,
    ) -> CountResult:
        """Count models and return a structured result."""
        if method == "enumeration":
            return self.count_model_by_bv_enumeration_result(variables=variables)
        if method in ("sharp_sat", "sharpsat"):
            return self.count_models_by_sharp_sat_result(variables=variables)
        if method in ("approx", "approxmc"):
            return self.count_models_by_approxmc_result(variables=variables)
        if method != "auto":
            return unsupported_count_result(
                backend="bv-none",
                reason=f"unknown bit-vector counting method: {method}",
            )

        if variables is not None:
            return self.count_model_by_bv_enumeration_result(variables=variables)

        total_bit_width = sum(_bit_width(var) for var in self.vars)
        if total_bit_width <= 16:
            result = self.count_model_by_bv_enumeration_result()
            result.metadata["selection"] = "auto"
            return result

        approx_result = self.count_models_by_approxmc_result()
        if approx_result.status != "unsupported":
            approx_result.metadata["selection"] = "auto"
            return approx_result

        result = self.count_model_by_bv_enumeration_result()
        result.metadata["selection"] = "auto-fallback"
        return result
