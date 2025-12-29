"""Preprocessing and Boolean abstraction for CDCL(T)"""

from typing import List
import z3
from aria.utils import SolverResult


def extract_literals_from_cnf(clauses: List) -> List[List]:
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
        self.numeric_clauses = []  # [[1, -2], [3, 4], ...]

        # Theory data
        self.theory_signature = []  # SMT-LIB2 variable declarations
        self.theory_constraints = ""  # Initial theory constraints

        # Boolean signature (for simple CDCL variant)
        self.bool_signature = []
        self.bool_constraints = ""

        self._next_var_id = 1

    def _make_bool_var(self, atom) -> z3.ExprRef:
        """Create or retrieve Boolean variable for a theory atom"""
        atom_str = str(atom)
        if atom_str in self.var_to_id:
            # Already abstracted
            var_name = self.id_to_var[self.var_to_id[atom_str]]
            return z3.Bool(var_name)

        # Create new Boolean variable
        var_name = f"p@{self._next_var_id}"
        var_id = self._next_var_id
        self._next_var_id += 1

        self.bool_var_names.append(var_name)
        self.var_to_id[var_name] = var_id
        self.id_to_var[var_id] = var_name

        return z3.Bool(var_name)

    def _abstract_literal(self, lit, atom_to_bool) -> z3.ExprRef:
        """Abstract a single literal"""
        if z3.is_not(lit):
            inner = lit.arg(0)
            if inner not in atom_to_bool:
                atom_to_bool[inner] = self._make_bool_var(inner)
            return z3.Not(atom_to_bool[inner])
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
                    numeric_clause.append(-self.var_to_id[var_name])
                else:
                    var_name = str(lit)
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
        # Parse and simplify
        fml = z3.And(z3.parse_smt2_string(smt2_string))

        # Apply preprocessing tactics
        tactics = z3.Then('simplify', 'elim-uncnstr', 'solve-eqs', 'simplify', 'tseitin-cnf')
        simplified = tactics(fml)
        result_expr = simplified.as_expr()

        # Check if we can decide immediately
        if z3.is_false(result_expr):
            return SolverResult.UNSAT
        if z3.is_true(result_expr):
            return SolverResult.SAT

        # Build Boolean abstraction
        cnf_clauses = extract_literals_from_cnf(simplified[0])
        atom_to_bool = {}
        bool_clauses = [self._abstract_clause(clause, atom_to_bool) for clause in cnf_clauses]

        # Extract signatures using Z3 solver
        solver = z3.Solver()
        solver.add(result_expr)
        solver.add(bool_clauses)

        sexpr = solver.sexpr()
        for line in sexpr.split("\n"):
            if line.startswith("(as"):
                break
            if "p@" in line:
                self.bool_signature.append(line)
            self.theory_signature.append(line)

        # Build constraints
        self.bool_constraints = z3.And(bool_clauses).sexpr()

        # Theory constraints: map each Boolean var to its theory atom
        theory_equiv = z3.And([bool_var == atom for atom, bool_var in atom_to_bool.items()])
        self.theory_constraints = theory_equiv.sexpr()

        # Build numeric clauses for SAT solver
        self._build_numeric_clauses(bool_clauses)

        return SolverResult.UNKNOWN
