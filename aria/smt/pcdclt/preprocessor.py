"""Preprocessing and Boolean abstraction for CDCL(T)"""

from typing import Any, List, cast
import time
import z3
from aria.utils import SolverResult

import logging
logger = logging.getLogger(__name__)

def extract_literals_from_cnf(clauses: List[Any]) -> List[List[Any]]:
    """Convert Z3 Or-expr list into CNF-like list-of-lists."""
    result = []
    for clause in clauses:
        if z3.is_or(clause):
            result.append(list(clause.children()))
        else:
            result.append([clause])
    return result


class FormulaAbstraction:
    """Manages Boolean abstraction and theory constraints"""

    def __init__(self):
        # Boolean abstraction data
        self.bool_var_names = []  # ['p@0', 'p@1', ...]
        self.var_to_id = {}  # 'p@0' -> 1
        self.id_to_var = {}  # 1 -> 'p@0'
        self.id_to_atom = {}
        self.id_to_atom_sexpr = {}
        self.numeric_clauses = []  # [[1, -2], [3, 4], ...]

        # Theory data
        self.theory_signature = []  # SMT-LIB2 variable declarations
        self.theory_constraints = ""  # Initial theory constraints

        # Boolean signature (for simple CDCL variant)
        self.bool_signature = []
        self.bool_constraints = ""

        self._next_var_id = 1
    def _make_bool_var(self, atom: Any) -> z3.ExprRef:
        """Create or retrieve Boolean variable for a theory atom"""
        # Create new Boolean variable
        var_name = f"p@{self._next_var_id}"
        var_id = self._next_var_id
        self._next_var_id += 1

        self.bool_var_names.append(var_name)
        self.var_to_id[var_name] = var_id
        self.id_to_var[var_id] = var_name
        self.id_to_atom[var_id] = atom

        bool_var = z3.Bool(var_name)
        return bool_var

    def get_atom_sexpr(self, var_id: int) -> str:
        if var_id not in self.id_to_atom_sexpr:
            self.id_to_atom_sexpr[var_id] = self.id_to_atom[var_id].sexpr()
        return self.id_to_atom_sexpr[var_id]

    def _abstract_formula(
        self,
        expr: Any,
        atom_to_bool: dict[Any, z3.ExprRef],
        memo: dict[Any, z3.ExprRef],
    ) -> Any:
        if z3.is_true(expr) or z3.is_false(expr):
            return cast(z3.ExprRef, expr)

        if expr in memo:
            return memo[expr]

        if expr.sort().kind() != z3.Z3_BOOL_SORT:
            return cast(z3.ExprRef, expr)

        decl_kind = expr.decl().kind() if hasattr(expr, "decl") else None

        if z3.is_not(expr):
            child = expr.arg(0)
            if child.sort().kind() == z3.Z3_BOOL_SORT and child.num_args() > 0:
                result = cast(
                    z3.ExprRef,
                    z3.Not(self._abstract_formula(child, atom_to_bool, memo)),
                )
            else:
                result = self._abstract_literal(expr, atom_to_bool)
            memo[expr] = result
            return result

        if z3.is_and(expr):
            result = cast(
                z3.ExprRef,
                z3.And(
                    [
                        self._abstract_formula(child, atom_to_bool, memo)
                        for child in expr.children()
                    ]
                ),
            )
        elif z3.is_or(expr):
            result = cast(
                z3.ExprRef,
                z3.Or(
                    [
                        self._abstract_formula(child, atom_to_bool, memo)
                        for child in expr.children()
                    ]
                ),
            )
        elif z3.is_implies(expr):
            lhs, rhs = expr.children()
            result = cast(
                z3.ExprRef,
                z3.Implies(
                    self._abstract_formula(lhs, atom_to_bool, memo),
                    self._abstract_formula(rhs, atom_to_bool, memo),
                ),
            )
        elif decl_kind == z3.Z3_OP_XOR:
            lhs, rhs = expr.children()
            result = cast(
                z3.ExprRef,
                z3.Xor(
                    self._abstract_formula(lhs, atom_to_bool, memo),
                    self._abstract_formula(rhs, atom_to_bool, memo),
                ),
            )
        elif decl_kind == z3.Z3_OP_ITE:
            cond, t_branch, f_branch = expr.children()
            result = cast(
                z3.ExprRef,
                z3.If(
                    self._abstract_formula(cond, atom_to_bool, memo),
                    self._abstract_formula(t_branch, atom_to_bool, memo),
                    self._abstract_formula(f_branch, atom_to_bool, memo),
                ),
            )
        else:
            result = self._abstract_literal(expr, atom_to_bool)

        memo[expr] = result
        return result

    def _abstract_literal(self, lit: Any, atom_to_bool: dict[Any, z3.ExprRef]) -> z3.ExprRef:
        """Abstract a single literal"""
        if z3.is_not(lit):
            inner = lit.arg(0)
            if inner not in atom_to_bool:
                atom_to_bool[inner] = self._make_bool_var(inner)
            return cast(z3.ExprRef, z3.Not(atom_to_bool[inner]))
        if lit not in atom_to_bool:
            atom_to_bool[lit] = self._make_bool_var(lit)
        return atom_to_bool[lit]

    def _abstract_clause(self, clause, atom_to_bool):
        """Abstract a clause"""
        lits = clause if isinstance(clause, list) else [clause]
        return z3.Or([self._abstract_literal(lit, atom_to_bool) for lit in lits])

    def _build_numeric_clauses(self, z3_clauses):
        """Convert Z3 Boolean clauses to numeric clauses for SAT solver"""
        for cls in z3_clauses:
            numeric_clause = []
            literals = cls.children() if z3.is_or(cls) else [cls]

            for lit in literals:
                if z3.is_not(lit):
                    var_name = str(lit.children()[0])
                else:
                    var_name = str(lit)

                if var_name not in self.var_to_id:
                    var_id = self._next_var_id
                    self._next_var_id += 1
                    self.bool_var_names.append(var_name)
                    self.var_to_id[var_name] = var_id
                    self.id_to_var[var_id] = var_name

                if z3.is_not(lit):
                    numeric_clause.append(-self.var_to_id[var_name])
                else:
                    numeric_clause.append(self.var_to_id[var_name])

            self.numeric_clauses.append(numeric_clause)

    def preprocess(self, smt2_string: str) -> SolverResult:
        """
        Preprocess SMT formula: simplify, convert to CNF, build Boolean abstraction

        Returns:
            SolverResult.SAT if formula is trivially SAT
            SolverResult.UNSAT if formula is trivially UNSAT
            SolverResult.UNKNOWN if further solving is needed
        """
        start = time.monotonic()

        logger.info("preprocess parse start", extra={"is_timing": True})
        parsed_assertions = z3.parse_smt2_string(smt2_string)
        logger.info(
            "preprocess parse over assertions=%d elapsed=%.3fs",
            len(parsed_assertions),
            time.monotonic() - start,
            extra={"is_timing": True},
        )

        fml = z3.And(parsed_assertions)
        atom_to_bool = {}
        memo = {}

        abstract_start = time.monotonic()
        logger.info("preprocess abstract start", extra={"is_timing": True})
        abstracted = self._abstract_formula(fml, atom_to_bool, memo)
        logger.info(
            "preprocess abstract over atoms=%d elapsed=%.3fs",
            len(atom_to_bool),
            time.monotonic() - abstract_start,
            extra={"is_timing": True},
        )

        tactic_start = time.monotonic()
        logger.info("preprocess tactic start", extra={"is_timing": True})
        simplified = z3.Then(
            z3.Tactic("simplify"),
            z3.Tactic("propagate-values"),
            z3.Tactic("tseitin-cnf"),
        )(abstracted)
        logger.info(
            "preprocess tactic over goals=%d elapsed=%.3fs",
            len(simplified),
            time.monotonic() - tactic_start,
            extra={"is_timing": True},
        )

        # for debugging, we keep it simple
        # SMT-COMP 2025 - parallel/QF_LIA/scrambled103783.smt2
        # simplified = z3.Tactic("simplify")(fml)
        # simplified = z3.Tactic("elim-uncnstr")(simplified.as_expr())
        # simplified = z3.Tactic("solve-eqs")(simplified.as_expr())
        # simplified = z3.Tactic("simplify")(simplified.as_expr())
        # # logger.debug(simplified.as_expr().sexpr())
        # simplified = z3.Tactic("tseitin-cnf")(simplified.as_expr())

        result_expr = simplified.as_expr()

        # Check if we can decide immediately
        if z3.is_false(result_expr):
            return SolverResult.UNSAT
        if z3.is_true(result_expr):
            return SolverResult.SAT

        # CNF is now purely propositional over abstraction variables and Tseitin variables.
        bool_clauses = list(cast(Any, simplified[0]))

        # Build constraints
        self.bool_constraints = cast(z3.ExprRef, z3.And(bool_clauses)).sexpr()

        # Build numeric clauses for SAT solver
        numeric_start = time.monotonic()
        logger.info("preprocess numeric start", extra={"is_timing": True})
        self._build_numeric_clauses(bool_clauses)
        logger.info(
            "preprocess numeric over clauses=%d elapsed=%.3fs total=%.3fs",
            len(self.numeric_clauses),
            time.monotonic() - numeric_start,
            time.monotonic() - start,
            extra={"is_timing": True},
        )

        return SolverResult.UNKNOWN
