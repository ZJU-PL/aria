"""Existential QE for Boolean CNFs via Shannon expansion.

This prototype handles only existential quantifier elimination on Boolean
formulas represented in CNF.
"""

from pysat.formula import CNF
from pysat.solvers import Glucose3


class QuantifierElimination:
    """Existential quantifier elimination for Boolean CNF formulas."""

    def __init__(self):
        """Initialize the quantifier elimination instance."""
        self.var_counter = 1
        self.var_map = {}
        self.reverse_map = {}

    def get_var_id(self, var_name):
        """Get or create a variable ID for the given variable name."""
        if var_name not in self.var_map:
            self.var_map[var_name] = self.var_counter
            self.reverse_map[self.var_counter] = var_name
            self.var_counter += 1
        return self.var_map[var_name]

    @staticmethod
    def _is_true(cnf_formula):
        """Return whether the CNF formula is logically true."""
        return len(cnf_formula.clauses) == 0

    @staticmethod
    def _is_false(cnf_formula):
        """Return whether the CNF formula is logically false."""
        return any(len(clause) == 0 for clause in cnf_formula.clauses)

    @staticmethod
    def _normalize_clause(clause):
        """Deduplicate literals and drop tautological clauses."""
        normalized = []
        seen = set()

        for lit in clause:
            if -lit in seen:
                return None
            if lit not in seen:
                seen.add(lit)
                normalized.append(lit)

        return normalized

    def shannon_expand(self, cnf_formula, var_id):
        """Perform Shannon expansion on a CNF formula for a given variable."""
        pos_cofactor, neg_cofactor = CNF(), CNF()

        for clause in cnf_formula.clauses:
            has_pos = var_id in clause
            has_neg = -var_id in clause
            new_clause = [lit for lit in clause if abs(lit) != var_id]

            if not has_pos:
                pos_cofactor.append(new_clause)
            if not has_neg:
                neg_cofactor.append(new_clause)

        return pos_cofactor, neg_cofactor

    def eliminate_exists(self, cnf_formula, var_id):
        """Eliminate one existentially quantified Boolean variable from a CNF."""
        pos_cofactor, neg_cofactor = self.shannon_expand(cnf_formula, var_id)

        if self._is_true(pos_cofactor) or self._is_true(neg_cofactor):
            return CNF()
        if self._is_false(pos_cofactor) and self._is_false(neg_cofactor):
            return CNF(from_clauses=[[]])
        if self._is_false(pos_cofactor):
            return neg_cofactor
        if self._is_false(neg_cofactor):
            return pos_cofactor

        result = CNF()
        seen_clauses = set()

        for clause1 in pos_cofactor.clauses:
            for clause2 in neg_cofactor.clauses:
                resolvent = self._normalize_clause(clause1 + clause2)

                if resolvent is None:
                    continue

                resolvent_key = tuple(sorted(resolvent))
                if resolvent_key not in seen_clauses:
                    seen_clauses.add(resolvent_key)
                    result.append(resolvent)

        return result

    def eliminate_quantifiers(self, cnf_formula, variables):
        """Eliminate a list of existentially quantified Boolean variables."""
        result = cnf_formula
        for var in variables:
            var_id = self.get_var_id(var)
            result = self.eliminate_exists(result, var_id)
        return result

    def is_satisfiable(self, cnf_formula):
        """Check if a CNF formula is satisfiable."""
        with Glucose3() as solver:
            solver.append_formula(cnf_formula)
            return solver.solve()


# Minimal example usage
if __name__ == "__main__":
    qe = QuantifierElimination()
    test_formula = CNF()
    x = qe.get_var_id("x")
    y = qe.get_var_id("y")
    test_formula.append([x, y])
    print(qe.eliminate_quantifiers(test_formula, ["x"]).clauses)
