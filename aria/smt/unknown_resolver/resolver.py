"""
SAE Unknown Formula Resolver - Systematic mutation-based resolution of unknown SMT formulas
"""

import z3
import subprocess
import tempfile
import os
import hashlib
from pathlib import Path
from typing import Dict, Set, List, Any, Tuple


class SAEUnknownResolver:
    """Resolves unknown SMT formulas using systematic mutation space coverage."""

    def __init__(self, solver_binary: str, timeout: int = 30, max_depth: int = 3,
                 max_mutations: int = 20, verbose: bool = False) -> None:
        self.solver_binary = Path(solver_binary)
        self.timeout = timeout
        self.max_depth = max_depth
        self.max_mutations = max_mutations
        self.verbose = verbose

        if not self.solver_binary.exists():
            raise FileNotFoundError(f"Solver binary not found: {solver_binary}")

        self.applied_mutations = {'structural': set(), 'pst': set()}
        self.mutation_combinations = set()
        self.stats = {'mutations_tried': 0, 'formulas_explored': 0}

    def resolve(self, formula: z3.ExprRef) -> str:
        if self.verbose:
            print(f"Resolving: {formula}")

        cnf = self._to_cnf(formula)

        # Try single mutations first
        result = self._try_single_mutations(cnf)
        if result != "unknown":
            return result

        # Try combinations
        result = self._try_combinations(cnf)
        if result != "unknown":
            return result

        return self._depth_search(cnf)

    def _to_cnf(self, formula: z3.ExprRef) -> z3.ExprRef:
        goal = z3.Goal()
        goal.add(formula)
        cnf_goal = z3.Tactic('tseitin-cnf')(goal)

        if not cnf_goal:
            return z3.BoolVal(True)

        assertions = []
        for subgoal in cnf_goal:
            assertions.extend([subgoal[i] for i in range(len(subgoal))])

        return z3.And(assertions) if assertions else z3.BoolVal(True)

    def _try_single_mutations(self, formula: z3.ExprRef) -> str:
        result = self._try_structural(formula)
        if result != "unknown":
            return result
        return self._try_pst(formula)

    def _try_structural(self, formula: z3.ExprRef) -> str:
        # Remove conjuncts
        if z3.is_and(formula):
            conjuncts = formula.children()
            for i, conjunct in enumerate(conjuncts):
                if self._is_cached('structural', conjunct):
                    continue

                remaining = conjuncts[:i] + conjuncts[i+1:]
                mutated = z3.BoolVal(True) if not remaining else (
                    remaining[0] if len(remaining) == 1 else z3.And(remaining))

                result = self._solve(mutated)
                self._cache_mutation('structural', conjunct)
                if result == "unsat":
                    return "unsat"

        # Remove disjuncts
        for clause_idx, clause in enumerate(self._extract_clauses(formula)):
            if z3.is_or(clause):
                disjuncts = clause.children()
                for lit_idx, literal in enumerate(disjuncts):
                    if self._is_cached('structural', literal) or len(disjuncts) <= 1:
                        continue

                    remaining = disjuncts[:lit_idx] + disjuncts[lit_idx+1:]
                    new_clause = remaining[0] if len(remaining) == 1 else z3.Or(remaining)
                    mutated = self._replace_clause(formula, clause_idx, new_clause)

                    result = self._solve(mutated)
                    self._cache_mutation('structural', literal)
                    if result == "sat":
                        return "sat"

        # Instantiate variables
        for var in list(self._get_variables(formula))[:3]:
            if self._is_cached('structural', var):
                continue

            for value in self._get_instantiation_values(var, formula):
                if self._solve(z3.substitute(formula, (var, value))) == "sat":
                    self._cache_mutation('structural', var)
                    return "sat"
            self._cache_mutation('structural', var)

        return "unknown"

    def _try_pst(self, formula: z3.ExprRef) -> str:
        for literal in self._get_literals(formula):
            if self._is_cached('pst', literal):
                continue

            # Try over-approximations (expect unsat)
            for new_literal in self._get_pst_mutations(literal, True):
                if self._solve(z3.substitute(formula, (literal, new_literal))) == "unsat":
                    self._cache_mutation('pst', literal)
                    return "unsat"

            # Try under-approximations (expect sat)
            for new_literal in self._get_pst_mutations(literal, False):
                if self._solve(z3.substitute(formula, (literal, new_literal))) == "sat":
                    self._cache_mutation('pst', literal)
                    return "sat"

            self._cache_mutation('pst', literal)

        return "unknown"

    def _try_combinations(self, formula: z3.ExprRef) -> str:
        promising = self._get_promising_mutations(formula)

        for i, (type1, mut1, desc1) in enumerate(promising):
            for type2, mut2, desc2 in promising[i+1:]:
                if type1 == type2:
                    continue

                combo_hash = self._hash(f"{desc1}|{desc2}")
                if combo_hash in self.mutation_combinations:
                    continue

                self.mutation_combinations.add(combo_hash)

                try:
                    if type1 == "over" and type2 == "under":
                        literals = self._get_literals(mut1)
                        if literals:
                            under_muts = self._get_pst_mutations(literals[0], False)
                            if under_muts:
                                combined = z3.substitute(mut1, (literals[0], under_muts[0]))
                                result = self._solve(combined)
                                if result in ["sat", "unsat"]:
                                    return result
                except:
                    continue

        return "unknown"

    def _depth_search(self, formula: z3.ExprRef) -> str:
        queue = [(formula, 0, [])]
        visited = set()

        while queue and self.stats['mutations_tried'] < self.max_mutations * 2:
            current, depth, path = queue.pop(0)

            if depth >= self.max_depth or self._hash(current) in visited:
                continue

            visited.add(self._hash(current))
            result = self._solve(current)
            if result != "unknown":
                return result

            for mutated, new_path in self._get_simple_mutations(current, path)[:5]:
                queue.append((mutated, depth + 1, new_path))
            queue = queue[:25]

        return "unknown"

    def _get_pst_mutations(self, literal: z3.ExprRef, is_over_approx: bool) -> List[z3.ExprRef]:
        if z3.is_not(literal):
            inner = literal.children()[0]
            inner_muts = self._get_pst_mutations(inner, not is_over_approx)
            return [z3.Not(m) for m in inner_muts]

        mutations = []
        if z3.is_arith(literal):
            args = literal.children()
            op = literal.decl().kind()

            if is_over_approx:
                if op == z3.Z3_OP_LT:
                    mutations.extend([args[0] <= args[1], args[0] != args[1]])
                elif op == z3.Z3_OP_GT:
                    mutations.extend([args[0] >= args[1], args[0] != args[1]])
                elif op == z3.Z3_OP_EQ:
                    mutations.extend([args[0] <= args[1], args[0] >= args[1]])
            else:
                if op == z3.Z3_OP_LT:
                    mutations.append(args[0] <= args[1] - 1)
                elif op == z3.Z3_OP_GT:
                    mutations.append(args[0] >= args[1] + 1)
                elif op == z3.Z3_OP_EQ:
                    mutations.append(z3.And(args[0] == z3.IntVal(0), args[1] == z3.IntVal(0)))

        return mutations[:2]

    def _get_instantiation_values(self, var: z3.ExprRef, formula: z3.ExprRef) -> List[z3.ExprRef]:
        values = []

        if z3.is_int(var):
            constants = self._extract_constants(formula, z3.is_int_value)
            values.extend(constants[:2])
            values.extend([z3.IntVal(0), z3.IntVal(1), z3.IntVal(-1)])
        elif z3.is_real(var):
            values.extend([z3.RealVal(0), z3.RealVal(1), z3.RealVal(-1)])
        elif z3.is_bv(var):
            size = var.sort().size()
            values.extend([z3.BitVecVal(0, size), z3.BitVecVal(1, size)])

        return values[:3]

    def _solve(self, formula: z3.ExprRef) -> str:
        try:
            smtlib = self._to_smtlib(formula)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                f.write(smtlib)
                temp_file = f.name

            try:
                result = subprocess.run(
                    [str(self.solver_binary), temp_file],
                    capture_output=True, text=True, timeout=self.timeout
                )
                output = result.stdout.strip().lower()
                self.stats['mutations_tried'] += 1

                if 'sat' in output and 'unsat' not in output:
                    return "sat"
                elif 'unsat' in output:
                    return "unsat"
                return "unknown"
            finally:
                os.unlink(temp_file)
        except:
            return "unknown"

    def _to_smtlib(self, formula: z3.ExprRef) -> str:
        variables = self._get_variables(formula)
        lines = ["(set-logic ALL)"]
        lines.extend([f"(declare-fun {var} () {var.sort()})" for var in variables])
        lines.extend([f"(assert {formula})", "(check-sat)", "(exit)"])
        return '\n'.join(lines)

    # Helper methods
    def _hash(self, obj: Any) -> str:
        return hashlib.md5(str(obj).encode()).hexdigest()

    def _is_cached(self, mutation_type: str, obj: Any) -> bool:
        return self._hash(obj) in self.applied_mutations[mutation_type]

    def _cache_mutation(self, mutation_type: str, obj: Any) -> None:
        self.applied_mutations[mutation_type].add(self._hash(obj))

    def _get_variables(self, formula: z3.ExprRef) -> Set[z3.ExprRef]:
        variables = set()
        def collect(expr: z3.ExprRef):
            if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                variables.add(expr)
            for child in expr.children():
                collect(child)
        collect(formula)
        return variables

    def _get_literals(self, formula: z3.ExprRef) -> List[z3.ExprRef]:
        literals = []
        def collect(expr: z3.ExprRef):
            if z3.is_and(expr):
                for child in expr.children():
                    collect(child)
            elif z3.is_or(expr):
                literals.extend(expr.children())
            else:
                literals.append(expr)
        collect(formula)
        return literals

    def _extract_clauses(self, formula: z3.ExprRef) -> List[z3.ExprRef]:
        return formula.children() if z3.is_and(formula) else [formula]

    def _replace_clause(self, formula: z3.ExprRef, idx: int, new_clause: z3.ExprRef) -> z3.ExprRef:
        if z3.is_and(formula):
            clauses = formula.children()
            new_clauses = clauses[:idx] + [new_clause] + clauses[idx+1:]
            return z3.And(new_clauses)
        return new_clause

    def _extract_constants(self, formula: z3.ExprRef, predicate: Any) -> List[z3.ExprRef]:
        constants = []
        def collect(expr: z3.ExprRef):
            if predicate(expr):
                constants.append(expr)
            for child in expr.children():
                collect(child)
        collect(formula)
        return list(set(constants))

    def _get_promising_mutations(self, formula: z3.ExprRef) -> List[Tuple[str, z3.ExprRef, str]]:
        promising = []
        if z3.is_and(formula):
            conjuncts = formula.children()
            for i, conjunct in enumerate(conjuncts[:3]):
                remaining = conjuncts[:i] + conjuncts[i+1:]
                if remaining:
                    mutated = z3.And(remaining) if len(remaining) > 1 else remaining[0]
                    if self._solve(mutated) == "unknown":
                        promising.append(("over", mutated, f"rm_conj_{i}"))
        return promising[:5]

    def _get_simple_mutations(self, formula: z3.ExprRef, path: List[str]) -> List[Tuple[z3.ExprRef, List[str]]]:
        mutations = []

        # Remove conjuncts
        if z3.is_and(formula):
            conjuncts = formula.children()
            for i, conjunct in enumerate(conjuncts[:2]):
                remaining = conjuncts[:i] + conjuncts[i+1:]
                if remaining:
                    mutated = z3.And(remaining) if len(remaining) > 1 else remaining[0]
                    mutations.append((mutated, path + [f"rm_{i}"]))

        # PST mutations
        for literal in self._get_literals(formula)[:2]:
            for new_literal in self._get_pst_mutations(literal, True)[:1]:
                mutated = z3.substitute(formula, (literal, new_literal))
                mutations.append((mutated, path + [f"pst_{literal}"]))

        return mutations

    def get_stats(self) -> Dict[str, int]:
        """Get mutation statistics."""
        return {
            'mutations_tried': self.stats['mutations_tried'],
            'formulas_explored': self.stats['formulas_explored'],
            'structural_mutations': len(self.applied_mutations['structural']),
            'pst_mutations': len(self.applied_mutations['pst'])
        }

# Usage example
if __name__ == "__main__":
    x, y = z3.Ints('x y')
    formula = z3.And(x > 0, y > 0, x + y < 5, x * y > 10)

    resolver = SAEUnknownResolver("/path/to/z3", verbose=True)
    result = resolver.resolve(formula)
    print(f"Result: {result}")
    print(f"Stats: {resolver.get_stats()}")
