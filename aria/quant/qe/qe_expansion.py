"""Quantifier Elimination for Boolean formulas via Shannon Expansion"""

from pysat.formula import CNF
from pysat.solvers import Glucose3


class QuantifierElimination:
    """Quantifier elimination using Shannon expansion for Boolean formulas."""

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

    def shannon_expand(self, cnf_formula, var_id):
        """Perform Shannon expansion on a CNF formula for a given variable."""
        pos_cofactor, neg_cofactor = CNF(), CNF()
        for clause in cnf_formula.clauses:
            new_clause = [lit for lit in clause if abs(lit) != var_id]
            if any(lit == var_id for lit in clause):
                if not any(lit > 0 and abs(lit) == var_id for lit in clause):
                    pos_cofactor.append(new_clause)
            else:
                pos_cofactor.append(new_clause)
            if any(lit == -var_id for lit in clause):
                if not any(lit < 0 and abs(lit) == var_id for lit in clause):
                    neg_cofactor.append(new_clause)
            else:
                neg_cofactor.append(new_clause)
        return pos_cofactor, neg_cofactor

    def eliminate_exists(self, cnf_formula, var_id):
        """Eliminate existential quantifier for a given variable."""
        pos_cofactor, neg_cofactor = self.shannon_expand(cnf_formula, var_id)
        result = CNF()
        if not pos_cofactor.clauses or not neg_cofactor.clauses:
            return CNF()
        for clause1 in pos_cofactor.clauses:
            for clause2 in neg_cofactor.clauses:
                resolvent = list(set(clause1 + clause2))
                if resolvent:
                    result.append(resolvent)
        return result

    def eliminate_quantifiers(self, cnf_formula, variables):
        """Eliminate quantifiers for a list of variables."""
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
