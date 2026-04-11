"""Boolean QE for CNFs via Shannon expansion.

This prototype handles Boolean formulas represented in CNF and supports both
existential and universal quantifier elimination by iterating one variable at a
time.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from pysat.formula import CNF
from pysat.solvers import Glucose3


VariableRef = Union[int, str]
QuantifierSpec = Tuple[str, Union[VariableRef, Sequence[VariableRef]]]


class QuantifierElimination:
    """Quantifier elimination for Boolean CNF formulas."""

    def __init__(self):
        """Initialize the quantifier elimination instance."""
        self.var_counter = 1
        self.var_map = {}
        self.reverse_map = {}

    def get_var_id(self, var_name: str) -> int:
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
    def _normalize_clause(clause: List[int]) -> Optional[List[int]]:
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

    @staticmethod
    def _simplify_cnf(cnf_formula: CNF) -> CNF:
        """Drop tautologies, duplicates, and subsumed clauses."""
        normalized_clauses = set()

        for clause in cnf_formula.clauses:
            normalized = QuantifierElimination._normalize_clause(clause)
            if normalized is None:
                continue
            normalized_clauses.add(tuple(sorted(normalized)))

        if () in normalized_clauses:
            return CNF(from_clauses=[[]])

        minimal_clauses = []
        for clause in sorted(normalized_clauses, key=len):
            clause_set = set(clause)
            if any(set(existing).issubset(clause_set) for existing in minimal_clauses):
                continue
            minimal_clauses.append(clause)

        return CNF(from_clauses=[list(clause) for clause in minimal_clauses])

    def _resolve_var_id(self, var: Union[int, str]) -> int:
        """Resolve either an integer variable ID or a variable name."""
        if isinstance(var, int):
            return var
        return self.get_var_id(var)

    def shannon_expand(self, cnf_formula: CNF, var_id: int) -> Tuple[CNF, CNF]:
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

        return self._simplify_cnf(pos_cofactor), self._simplify_cnf(neg_cofactor)

    def eliminate_exists(self, cnf_formula: CNF, var_id: int) -> CNF:
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

        return self._simplify_cnf(result)

    def eliminate_forall(self, cnf_formula: CNF, var_id: int) -> CNF:
        """Eliminate one universally quantified Boolean variable from a CNF."""
        pos_cofactor, neg_cofactor = self.shannon_expand(cnf_formula, var_id)

        result = CNF()
        for clause in pos_cofactor.clauses:
            result.append(clause)
        for clause in neg_cofactor.clauses:
            result.append(clause)

        return self._simplify_cnf(result)

    def eliminate_quantifiers(
        self,
        cnf_formula: CNF,
        variables: Iterable[Union[int, str]],
        quantifier: str = "exists",
    ) -> CNF:
        """Eliminate a block of identical Boolean quantifiers from a CNF."""
        result = cnf_formula

        if quantifier not in {"exists", "forall"}:
            raise ValueError("quantifier must be 'exists' or 'forall'")

        for var in variables:
            var_id = self._resolve_var_id(var)
            if quantifier == "exists":
                result = self.eliminate_exists(result, var_id)
            else:
                result = self.eliminate_forall(result, var_id)

        return result

    def eliminate_quantifier_blocks(
        self, cnf_formula: CNF, quantifiers: Iterable[QuantifierSpec]
    ) -> CNF:
        """Eliminate an ordered list of quantified Boolean variable blocks."""
        result = cnf_formula
        for quantifier, vars_in_block in quantifiers:
            if isinstance(vars_in_block, (str, int)):
                variables = [vars_in_block]
            else:
                variables = list(vars_in_block)
            result = self.eliminate_quantifiers(
                result, variables, quantifier=quantifier
            )
        return result

    def is_satisfiable(self, cnf_formula: CNF) -> bool:
        """Check if a CNF formula is satisfiable."""
        with Glucose3() as solver:
            solver.append_formula(cnf_formula)
            return bool(solver.solve())


# Minimal example usage
if __name__ == "__main__":
    qe = QuantifierElimination()
    test_formula = CNF()
    x = qe.get_var_id("x")
    y = qe.get_var_id("y")
    test_formula.append([x, y])
    print(qe.eliminate_quantifiers(test_formula, ["x"]).clauses)
