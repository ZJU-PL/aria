"""QBF parsing helpers and a lightweight Z3-backed QBF model."""

import functools
from typing import Dict, List, Optional, Sequence, Tuple

import z3
from pysat.solvers import Solver
from z3 import And, Bool, BoolVal, Not, Or, is_true

from .qdimacs_parser import parse_qdimacs_string


def foldr(func, acc, xs):
    """Fold right: apply function from right to left over list."""

    return functools.reduce(lambda x, y: func(y, x), xs[::-1], acc)


def z3_val_to_int(z3_val: z3.BoolRef) -> int:
    """Convert Z3 boolean value to integer (1 for True, 0 for False)."""

    return 1 if is_true(z3_val) else 0


def int_vec_to_z3(int_vec: Sequence[int]) -> List[z3.BoolRef]:
    """Convert integer vector to Z3 boolean values."""

    return [BoolVal(value == 1) for value in int_vec]


q_to_z3 = {1: z3.ForAll, -1: z3.Exists}


def _assign_literal(clauses: Sequence[Sequence[int]], literal: int) -> List[List[int]]:
    """Simplify clauses under a single literal assignment."""

    simplified: List[List[int]] = []
    negated_literal = -literal
    for clause in clauses:
        if literal in clause:
            continue
        reduced_clause = [term for term in clause if term != negated_literal]
        simplified.append(reduced_clause)
    return simplified


def _active_prefix(
    prefix_blocks: Sequence[Tuple[str, Sequence[int]]], clauses: Sequence[Sequence[int]]
) -> List[Tuple[str, List[int]]]:
    active_variables = {abs(literal) for clause in clauses for literal in clause}
    filtered: List[Tuple[str, List[int]]] = []
    for kind, variables in prefix_blocks:
        kept = [int(variable) for variable in variables if int(variable) in active_variables]
        if kept:
            filtered.append((kind, kept))
    return filtered


def _is_propositionally_satisfiable(
    clauses: Sequence[Sequence[int]], solver_name: str
) -> bool:
    with Solver(name=solver_name, bootstrap_with=[list(clause) for clause in clauses]) as solver:
        return solver.solve()


def _solve_qdimacs_recursively(
    prefix_blocks: Sequence[Tuple[str, Sequence[int]]],
    clauses: Sequence[Sequence[int]],
    solver_name: str,
) -> bool:
    if any(len(clause) == 0 for clause in clauses):
        return False
    if not clauses:
        return True

    active_prefix = _active_prefix(prefix_blocks, clauses)
    if not active_prefix:
        return _is_propositionally_satisfiable(clauses, solver_name)

    if not _is_propositionally_satisfiable(clauses, solver_name):
        return False

    quantifier, variables = active_prefix[0]
    branch_variable = variables[0]
    remaining_prefix = [(quantifier, variables[1:])] + active_prefix[1:]

    positive_branch = _assign_literal(clauses, branch_variable)
    negative_branch = _assign_literal(clauses, -branch_variable)

    if quantifier == "e":
        return _solve_qdimacs_recursively(
            remaining_prefix, positive_branch, solver_name
        ) or _solve_qdimacs_recursively(remaining_prefix, negative_branch, solver_name)

    return _solve_qdimacs_recursively(
        remaining_prefix, positive_branch, solver_name
    ) and _solve_qdimacs_recursively(remaining_prefix, negative_branch, solver_name)


class QDIMACSParser:
    """Parser for QDIMACS format strings returning a Z3-based QBF."""

    def __init__(self):
        self.vars: Dict[int, Bool] = {}
        self.num_vars = 0
        self.num_clauses = 0

    def get_var(self, var_id: int) -> Bool:
        """Get or create a Z3 Bool variable for a given DIMACS variable ID."""

        abs_id = abs(var_id)
        if abs_id not in self.vars:
            self.vars[abs_id] = Bool(f"x_{abs_id}")
        return self.vars[abs_id]

    def parse_qdimacs(self, qdimacs_str: str) -> "QBF":
        """Parse a QDIMACS format string and return a QBF object."""

        parsed = parse_qdimacs_string(qdimacs_str)
        self.num_vars = parsed.num_vars
        self.num_clauses = parsed.num_clauses

        q_list: List[Tuple[int, List[Bool]]] = []
        for kind, variables in parsed.parsed_prefix:
            quant_type = 1 if kind == "a" else -1
            q_list.append((quant_type, [self.get_var(variable) for variable in variables]))

        clauses = []
        for clause in parsed.clauses:
            if not clause:
                clauses.append(BoolVal(False))
                continue
            literals = []
            for literal in clause:
                var = self.get_var(abs(literal))
                literals.append(Not(var) if literal < 0 else var)
            clauses.append(Or(*literals))

        prop_formula = And(clauses) if clauses else BoolVal(True)
        return QBF(
            prop_formula,
            q_list,
            prefix_blocks=parsed.parsed_prefix,
            matrix_clauses=parsed.clauses,
        )


class QBF:
    """Quantified Boolean formula represented as a Z3 body plus prenex prefix."""

    def __init__(
        self,
        prop_formula,
        q_list=None,
        prefix_blocks: Optional[Sequence[Tuple[str, Sequence[int]]]] = None,
        matrix_clauses: Optional[Sequence[Sequence[int]]] = None,
    ):
        if q_list is None:
            q_list = []
        self._q_list = list(q_list)
        self._prop = prop_formula
        self._prefix_blocks = (
            [(kind, [int(variable) for variable in variables]) for kind, variables in prefix_blocks]
            if prefix_blocks is not None
            else None
        )
        self._matrix_clauses = (
            [[int(literal) for literal in clause] for clause in matrix_clauses]
            if matrix_clauses is not None
            else None
        )

    def get_prop(self):
        """Get the propositional body."""

        return self._prop

    def get_q_list(self):
        """Get the quantifier prefix."""

        return list(self._q_list)

    def get_prefix_blocks(self) -> Optional[List[Tuple[str, List[int]]]]:
        """Get the QDIMACS-style prefix when available."""

        if self._prefix_blocks is None:
            return None
        return [(kind, list(variables)) for kind, variables in self._prefix_blocks]

    def get_matrix_clauses(self) -> Optional[List[List[int]]]:
        """Get the CNF matrix clauses when available."""

        if self._matrix_clauses is None:
            return None
        return [list(clause) for clause in self._matrix_clauses]

    def quantifier_depth(self) -> int:
        """Return the number of prenex blocks."""

        return len(self._q_list)

    def quantified_variables(self) -> List[Bool]:
        """Return the quantified variables in prefix order."""

        return [variable for _, variables in self._q_list for variable in variables]

    def quantifier_prefix_summary(self) -> Dict[str, int]:
        """Summarize the prefix by quantifier kind."""

        return {
            "forall_blocks": sum(1 for kind, _ in self._q_list if kind == 1),
            "exists_blocks": sum(1 for kind, _ in self._q_list if kind == -1),
            "forall_vars": sum(
                len(variables) for kind, variables in self._q_list if kind == 1
            ),
            "exists_vars": sum(
                len(variables) for kind, variables in self._q_list if kind == -1
            ),
        }

    def to_z3(self):
        """Convert to a Z3 formula."""

        return foldr(
            lambda q_v, formula: q_to_z3[q_v[0]](q_v[1], formula),
            self._prop,
            self._q_list,
        )

    def negate(self):
        """Negate the formula and flip quantifier kinds."""

        new_q_list = [(-kind, variables) for (kind, variables) in self._q_list]
        return QBF(
            self._prop.children()[0] if z3.is_not(self._prop) else z3.Not(self._prop),
            new_q_list,
        )

    def well_named(self):
        """Check if all quantified variables are unique."""

        appeared = set()
        for _, var_vec in self._q_list:
            for variable in var_vec:
                name = str(variable)
                if name in appeared:
                    return False
                appeared.add(name)
        return True

    def solve(self, backend: str = "z3", solver_name: str = "m22") -> z3.CheckSatResult:
        """Solve the quantified Boolean formula with the selected backend."""

        if backend == "z3":
            solver = z3.Solver()
            solver.add(self.to_z3())
            return solver.check()
        if backend == "pysat":
            return self.solve_pysat(solver_name=solver_name)
        raise ValueError(f"unsupported QBF backend: {backend}")

    def solve_pysat(self, solver_name: str = "m22") -> z3.CheckSatResult:
        """Solve a prenex CNF QBF without falling back to non-PySAT engines."""

        if self._prefix_blocks is None or self._matrix_clauses is None:
            raise NotImplementedError(
                "PySAT QBF solving requires a prenex CNF/QDIMACS representation."
            )

        is_true = _solve_qdimacs_recursively(
            self._prefix_blocks, self._matrix_clauses, solver_name
        )
        return z3.sat if is_true else z3.unsat

    def solve_with_skolem(self):
        """Placeholder for future certificate/skolem extraction."""

        raise NotImplementedError(
            "Skolem extraction is not implemented; this method was unsound."
        )

    def __eq__(self, other):
        return self._prop.eq(other.get_prop()) and self._q_list == other.get_q_list()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        normalized_prefix = tuple(
            (kind, tuple(str(variable) for variable in variables))
            for kind, variables in self._q_list
        )
        return hash((hash(self._prop), normalized_prefix))


def demo():
    """Demo function to test QDIMACS parsing and solving."""

    qdimacs_str = """
    c Example QDIMACS file
    p cnf 4 2
    a 1 2 0
    e 3 4 0
    1 2 3 0
    -1 -2 4 0
    """

    parser = QDIMACSParser()
    qbf = parser.parse_qdimacs(qdimacs_str)
    result = qbf.solve()
    print(f"QBF Check: {result}")


if __name__ == "__main__":
    demo()
