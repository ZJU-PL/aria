"""
OMT(BV) solver using bit-blasting and MaxSAT reduction.
"""
import logging
import random
import time
from typing import Dict, List, Optional

import z3
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver

from arlib.bool.maxsat.maxsat_solver import MaxSATSolver
from arlib.smt.bv.mapped_blast import translate_smt2formula_to_cnf
from arlib.utils.z3_expr_utils import get_expr_vars

logger = logging.getLogger(__name__)

SAT_SOLVERS_IN_PYSAT = [
    'cd', 'cd15', 'gc3', 'gc4', 'g3',
    'g4', 'lgl', 'mcb', 'mpl', 'mg3',
    'mc', 'm22', 'msh'
]


class BitBlastOMTBVSolver:
    """OMT(BV) solver using bit-blasting and MaxSAT reduction."""

    def __init__(self) -> None:
        """Initialize the bit-blast OMT(BV) solver."""
        self.fml: Optional[z3.BoolRef] = None
        # Map a bit-vector variable to a list of Boolean variables (ordered by bit)
        self.bv2bool: Dict[str, List[str]] = {}
        # Map a Boolean variable to its internal ID in pysat
        self.bool2id: Dict[str, int] = {}
        self.vars: List[z3.ExprRef] = []
        self.verbose: int = 0
        self.engine: str = "FM"

    def from_smt_formula(self, formula: z3.BoolRef) -> None:
        """Set the SMT formula to optimize.

        Args:
            formula: Z3 boolean formula to optimize
        """
        self.fml = formula
        self.vars = get_expr_vars(self.fml)

    def set_engine(self, solver_name: str) -> None:
        """Set the MaxSAT solver engine to use.

        Args:
            solver_name: Name of the MaxSAT solver engine (e.g., "FM", "RC2", "OBV-BS")
        """
        self.engine = solver_name

    def bit_blast(self) -> List[List[int]]:
        """Convert a bit-vector formula to Boolean logic (CNF).

        Sets the `bv2bool` and `bool2id` class attributes as mappings from BV
        variables to boolean expressions and from boolean expressions to numerical IDs.

        Note: To track the correlation between bit-vector and Boolean variables,
        we use very restrictive pre-processing tactics in translate_smt2formula_to_cnf.
        Unfortunately, the translated Boolean formula can be very complex.
        In the OMT engine inside z3, more aggressive pre-processing can be used.
        However, we need a method to track the relation. For example, to encode a
        4-bit bit-vector x, aggressive pre-processing may result in a Boolean
        formula with only three variables {b1, b2, b3}. We don't know which
        bi corresponds to which bit of x.

        Returns:
            List of CNF clauses, where each clause is a list of integer literals
        """
        logger.debug("Start translating to CNF...")
        bv2bool, id_table, header, clauses = translate_smt2formula_to_cnf(self.fml)
        self.bv2bool = bv2bool
        self.bool2id = id_table
        logger.debug("  from bv to bools: %s", self.bv2bool)
        logger.debug("  from bool to pysat id: %s", self.bool2id)

        clauses_numeric = []
        for cls in clauses:
            clauses_numeric.append([int(lit) for lit in cls.split(" ")])
        return clauses_numeric

    def check_sat(self) -> Optional[bool]:
        """Check satisfiability of the SMT formula.

        Converts the formula to CNF and uses a SAT solver to check satisfiability.

        Returns:
            True if satisfiable, False if unsatisfiable, None if an error occurred

        TODO:
            Map back to bit-vector model
        """
        clauses_numeric = self.bit_blast()
        cnf = CNF(from_clauses=clauses_numeric)
        name = random.choice(SAT_SOLVERS_IN_PYSAT)
        try:
            start = time.time()
            with Solver(name=name, bootstrap_with=cnf) as solver:
                res = solver.solve()
                logger.debug("outcome by %s: %s", name, res)
            logger.debug("SAT solving time: %.3f", time.time() - start)
            return res
        except Exception as ex:
            logger.error("Error in SAT solving: %s", ex)
            return None

    def maximize_with_maxsat(self, obj: z3.ExprRef, is_signed: bool = False, minimize: bool = False) -> Optional[int]:
        """Solve OMT(BV) using MaxSAT reduction.

        Args:
            obj: Bit-vector expression to optimize
            is_signed: Interpret as signed (True) or unsigned (False)
            minimize: Minimize (True) or maximize (False)

        Returns:
            Optimal value or None if unsatisfiable
        """
        assert z3.is_bv(obj)

        # First check if the hard constraints are satisfiable using Z3
        solver = z3.Solver()
        solver.add(self.fml)
        sat_result = solver.check()
        if sat_result == z3.unsat:
            logger.debug("the hard formula is unsatisfiable")
            return None
        elif sat_result == z3.unknown:
            logger.warning("error checking satisfiability of hard formula")
            return None

        objname = obj

        if obj not in self.vars:
            objvars = get_expr_vars(obj)
            for v in objvars:
                if v not in self.vars:
                    raise ValueError(f"{obj} contains a variable not in the hard formula")
            # Create a new variable to represent obj (a term, e.g., x + y)
            objname = z3.BitVec(str(obj), objvars[0].sort().size())
            self.fml = z3.And(self.fml, objname == obj)
            self.vars.append(objname)

        after_simp = z3.Tactic("simplify")(self.fml).as_expr()
        if z3.is_true(after_simp):
            logger.debug("the hard formula is a tautology (obj can be any value)")
            return None
        elif z3.is_false(after_simp):
            logger.error("the hard formula with objective is trivially unsat")
            return None

        # For minimization of unsigned BVs, maximize ~obj instead
        # since maximizing ~x minimizes x (for unsigned interpretation)
        if minimize and not is_signed:
            not_obj = z3.BitVec(f"not_{str(objname)}", objname.size())
            self.fml = z3.And(self.fml, not_obj == ~objname)
            self.vars.append(not_obj)
            opt_obj = not_obj
            opt_obj_str = str(not_obj)
            do_minimize = False  # Now we're maximizing ~obj
        else:
            opt_obj = objname
            opt_obj_str = str(objname)
            do_minimize = minimize

        logger.debug("Start solving OMT(BV) by reducing to weighted Max-SAT...")
        clauses_numeric = self.bit_blast()
        wcnf = WCNF()
        wcnf.extend(clauses_numeric)
        total_score = 0
        bool_vars = self.bv2bool[opt_obj_str]
        num_bits = len(bool_vars)

        logger.debug("Start solving weighted Max-SAT via pySAT...")

        if is_signed:
            # Signed interpretation: MSB is sign bit
            weight_sign = -1 if do_minimize else 1
            for i in range(num_bits - 1):
                weight = weight_sign * (2 ** i)
                wcnf.append([self.bool2id[bool_vars[i]]], weight=weight)
                total_score += weight
            # Sign bit: 1 means negative, 0 means positive
            sign_bit_weight = weight_sign * (-(2 ** (num_bits - 1)))
            wcnf.append([self.bool2id[bool_vars[num_bits - 1]]], weight=sign_bit_weight)
            total_score += sign_bit_weight
        else:
            # Unsigned interpretation
            for i in range(num_bits):
                weight = 2 ** i
                wcnf.append([self.bool2id[bool_vars[i]]], weight=weight)
                total_score += weight

        maxsat_sol = MaxSATSolver(wcnf)
        maxsat_sol.set_maxsat_engine(self.engine)

        return self._solve_with_engine(maxsat_sol, opt_obj_str, total_score, bool_vars, is_signed)

    def _solve_with_engine(
        self,
        maxsat_sol: MaxSATSolver,
        obj_str: str,
        total_score: int,
        bool_vars: List[str],
        is_signed: bool,
    ) -> int:
        """Solve MaxSAT using the configured engine.

        Args:
            maxsat_sol: MaxSAT solver instance
            obj_str: String representation of the objective
            total_score: Total possible score
            bool_vars: List of boolean variable names
            is_signed: Whether to interpret as signed

        Returns:
            Maximum value found
        """
        if self.engine in ("FM", "RC2"):
            return self._solve_weighted(maxsat_sol, obj_str, total_score)
        elif self.engine == "OBV-BS":
            return self._solve_obv_bs(maxsat_sol, obj_str, bool_vars, is_signed)
        else:
            # Default to FM
            logger.warning("Unknown engine %s, defaulting to FM", self.engine)
            maxsat_sol.set_maxsat_engine("FM")
            return self._solve_weighted(maxsat_sol, obj_str, total_score)

    def _solve_weighted(self, maxsat_sol: MaxSATSolver, obj_str: str, total_score: int) -> int:
        """Solve using weighted MaxSAT (FM or RC2)."""
        start = time.time()
        maxsat_result = maxsat_sol.solve()
        cost = maxsat_result.cost
        result = total_score - cost
        # If we maximized ~obj for minimization, result is ~min_value, so compute ~result
        if obj_str.startswith("not_"):
            bv_width = len(self.bv2bool[obj_str[4:]])  # Remove "not_" prefix
            mask = (1 << bv_width) - 1
            result = mask ^ result
        logger.debug("MaxSAT cost: %d, total_score: %d", cost, total_score)
        logger.debug("optimal value of %s: %d", obj_str, result)
        logger.debug("%s MaxSAT time: %.3f", self.engine, time.time() - start)
        return result

    def _solve_obv_bs(
        self,
        maxsat_sol: MaxSATSolver,
        obj_str: str,
        bool_vars: List[str],
        is_signed: bool,
    ) -> int:
        """Solve using OBV-BS (binary-search-based) engine."""
        start = time.time()
        assumption_lits = maxsat_sol.solve()
        assumption_lits.reverse()
        sum_score = 0
        num_bits = len(assumption_lits)

        if is_signed:
            # Signed: process all bits except sign bit
            for i in range(num_bits - 1):
                if assumption_lits[i] > 0:
                    sum_score += 2 ** i
            # Sign bit: 1 means negative, 0 means positive
            if assumption_lits[-1] > 0:
                sum_score = -sum_score
        else:
            # Unsigned: process all bits
            for i in range(num_bits):
                if assumption_lits[i] > 0:
                    sum_score += 2 ** i

        logger.debug("maximum of %s: %d", obj_str, sum_score)
        logger.debug("OBV-BS MaxSAT time: %.3f", time.time() - start)
        return sum_score
