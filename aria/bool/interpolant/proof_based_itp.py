"""
Resolution proofs and proof-based Craig interpolation for Boolean formulas.

This module implements a small propositional resolution engine together with
McMillan's interpolation algorithm over the generated proof DAG. The focus is
correctness and a simple API, not proof-search performance.

The implementation supports two entry points:

- `BooleanInterpolant.compute_itp(fml_a, fml_b)` for Z3 Boolean formulas
- `BooleanInterpolant.compute_itp_from_cnf(fml_a, fml_b)` for PySAT CNFs

For non-CNF Boolean formulas, each side is clausified separately using local
Tseitin variables. Since those auxiliary variables are side-local, the extracted
interpolant only mentions atoms shared by the original inputs.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import z3
from pysat.formula import CNF

Clause = Tuple[int, ...]


def _mk_and(exprs: Sequence[z3.ExprRef]) -> z3.ExprRef:
    """Build a conjunction and handle the empty/singleton cases explicitly."""
    if not exprs:
        return z3.BoolVal(True)
    if len(exprs) == 1:
        return exprs[0]
    return z3.And(*exprs)


def _mk_or(exprs: Sequence[z3.ExprRef]) -> z3.ExprRef:
    """Build a disjunction and handle the empty/singleton cases explicitly."""
    if not exprs:
        return z3.BoolVal(False)
    if len(exprs) == 1:
        return exprs[0]
    return z3.Or(*exprs)


def _normalize_clause(literals: Sequence[int]) -> Optional[Clause]:
    """
    Canonicalize a clause.

    Returns `None` for tautological clauses that contain both `x` and `-x`.
    """
    literal_set = set()
    for literal in literals:
        if -literal in literal_set:
            return None
        literal_set.add(literal)

    return tuple(sorted(literal_set, key=lambda literal: (abs(literal), literal < 0)))


def _resolve_clauses(
    clause_a: Clause, clause_b: Clause, pivot: int
) -> Optional[Clause]:
    """Resolve two clauses on `pivot` if possible."""
    if pivot <= 0:
        raise ValueError("pivot must be a positive variable identifier")

    if pivot in clause_a and -pivot in clause_b:
        left = [literal for literal in clause_a if literal != pivot]
        right = [literal for literal in clause_b if literal != -pivot]
        return _normalize_clause(left + right)

    if -pivot in clause_a and pivot in clause_b:
        left = [literal for literal in clause_a if literal != -pivot]
        right = [literal for literal in clause_b if literal != pivot]
        return _normalize_clause(left + right)

    return None


class ClausePartition(Enum):
    """Origin partition of an initial clause."""

    A = "A"
    B = "B"


@dataclass(frozen=True)
class ResolutionStep:
    """One node of a resolution proof DAG."""

    clause: Clause
    partition: Optional[ClausePartition] = None
    left: Optional[int] = None
    right: Optional[int] = None
    pivot: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        """Whether this step is an initial clause."""
        return self.partition is not None


@dataclass
class FormulaEncoding:
    """CNF encoding of one partition."""

    clauses: List[Clause]
    vars_in_formula: Set[int]
    atom_vars: Set[int]


@dataclass
class ResolutionProof:
    """A resolution proof plus the metadata needed for interpolation."""

    steps: List[ResolutionStep]
    root: int
    var_to_expr: Dict[int, z3.ExprRef]
    vars_a: Set[int]
    vars_b: Set[int]

    @property
    def root_clause(self) -> Clause:
        """Return the derived root clause."""
        return self.steps[self.root].clause

    def validate(self) -> bool:
        """Check that every derived node is a valid resolution step."""
        for index, step in enumerate(self.steps):
            if step.is_leaf:
                if (
                    step.left is not None
                    or step.right is not None
                    or step.pivot is not None
                ):
                    return False
                continue

            if (
                step.left is None
                or step.right is None
                or step.pivot is None
                or step.left >= index
                or step.right >= index
            ):
                return False

            resolvent = _resolve_clauses(
                self.steps[step.left].clause,
                self.steps[step.right].clause,
                step.pivot,
            )
            if resolvent != step.clause:
                return False

        return self.root_clause == ()

    def extract_interpolant(self) -> z3.ExprRef:
        """
        Extract a Craig interpolant using McMillan's proof annotation rules.
        """
        shared_vars = self.vars_a & self.vars_b
        a_local_vars = self.vars_a - self.vars_b
        annotations: Dict[int, z3.ExprRef] = {}

        for index, step in enumerate(self.steps):
            if step.partition == ClausePartition.A:
                shared_literals = [
                    self._literal_to_expr(literal)
                    for literal in step.clause
                    if abs(literal) in shared_vars
                ]
                annotations[index] = z3.simplify(_mk_or(shared_literals))
                continue

            if step.partition == ClausePartition.B:
                annotations[index] = z3.BoolVal(True)
                continue

            if step.left is None or step.right is None or step.pivot is None:
                raise ValueError("derived proof node is missing resolution metadata")

            left_annotation = annotations[step.left]
            right_annotation = annotations[step.right]
            if step.pivot in a_local_vars:
                annotations[index] = z3.simplify(
                    z3.Or(left_annotation, right_annotation)
                )
            else:
                annotations[index] = z3.simplify(
                    z3.And(left_annotation, right_annotation)
                )

        return z3.simplify(annotations[self.root])

    def _literal_to_expr(self, literal: int) -> z3.ExprRef:
        """Convert an integer literal back to a Z3 Boolean expression."""
        atom = self.var_to_expr[abs(literal)]
        if literal < 0:
            return z3.Not(atom)
        return atom


class PropositionalClausifier:
    """
    Definitional CNF conversion with partition-local Tseitin variables.

    Original Boolean atoms are shared across both partitions by structural
    identity (`sexpr()`), while fresh definition variables are always local to
    one side.
    """

    def __init__(self):
        self._next_var = 1
        self._atom_to_var: Dict[str, int] = {}
        self.var_to_expr: Dict[int, z3.ExprRef] = {}

    def clausify(self, expr: z3.ExprRef, side: ClausePartition) -> FormulaEncoding:
        """Convert a Boolean formula to CNF clauses."""
        if not z3.is_bool(expr):
            raise TypeError("proof-based interpolation expects Boolean formulas")

        clauses: List[Clause] = []
        cache: Dict[str, int] = {}
        atom_vars: Set[int] = set()
        lowered = self._lower_boolean_formula(expr)
        root_literal = self._encode_literal(lowered, side, clauses, cache, atom_vars)
        root_clause = _normalize_clause([root_literal])
        if root_clause is None:
            raise ValueError("root literal unexpectedly produced a tautological clause")
        clauses.append(root_clause)

        vars_in_formula = set()
        for clause in clauses:
            vars_in_formula.update(abs(literal) for literal in clause)

        return FormulaEncoding(
            clauses=clauses, vars_in_formula=vars_in_formula, atom_vars=atom_vars
        )

    def _lower_boolean_formula(self, expr: z3.ExprRef) -> z3.ExprRef:
        """
        Rewrite Boolean connectives into the {And, Or, Not} basis.

        Non-Boolean theory atoms are treated as propositional atoms.
        """
        if z3.is_true(expr) or z3.is_false(expr):
            return expr

        if not z3.is_bool(expr):
            raise TypeError("expected a Boolean expression")

        kind = expr.decl().kind()
        if kind == z3.Z3_OP_NOT:
            return z3.Not(self._lower_boolean_formula(expr.arg(0)))

        if kind == z3.Z3_OP_AND:
            return z3.And(
                *[self._lower_boolean_formula(child) for child in expr.children()]
            )

        if kind == z3.Z3_OP_OR:
            return z3.Or(
                *[self._lower_boolean_formula(child) for child in expr.children()]
            )

        if kind == z3.Z3_OP_IMPLIES:
            lhs = self._lower_boolean_formula(expr.arg(0))
            rhs = self._lower_boolean_formula(expr.arg(1))
            return z3.Or(z3.Not(lhs), rhs)

        if kind == z3.Z3_OP_ITE and all(z3.is_bool(arg) for arg in expr.children()):
            cond = self._lower_boolean_formula(expr.arg(0))
            on_true = self._lower_boolean_formula(expr.arg(1))
            on_false = self._lower_boolean_formula(expr.arg(2))
            return z3.Or(z3.And(cond, on_true), z3.And(z3.Not(cond), on_false))

        if kind == z3.Z3_OP_EQ and all(z3.is_bool(arg) for arg in expr.children()):
            lowered_children = [
                self._lower_boolean_formula(child) for child in expr.children()
            ]
            if len(lowered_children) <= 1:
                return z3.BoolVal(True)
            equalities = [
                z3.Or(
                    z3.And(lowered_children[index], lowered_children[index + 1]),
                    z3.And(
                        z3.Not(lowered_children[index]),
                        z3.Not(lowered_children[index + 1]),
                    ),
                )
                for index in range(len(lowered_children) - 1)
            ]
            return _mk_and(equalities)

        if kind == z3.Z3_OP_DISTINCT and all(
            z3.is_bool(arg) for arg in expr.children()
        ):
            lowered_children = [
                self._lower_boolean_formula(child) for child in expr.children()
            ]
            return self._lower_distinct(lowered_children)

        if kind == z3.Z3_OP_XOR:
            lowered_children = [
                self._lower_boolean_formula(child) for child in expr.children()
            ]
            return self._lower_xor(lowered_children)

        return expr

    def _lower_xor(self, exprs: Sequence[z3.ExprRef]) -> z3.ExprRef:
        """Lower XOR into And/Or/Not."""
        if not exprs:
            return z3.BoolVal(False)

        parity = exprs[0]
        for expr in exprs[1:]:
            parity = z3.Or(
                z3.And(parity, z3.Not(expr)),
                z3.And(z3.Not(parity), expr),
            )
        return parity

    def _lower_distinct(self, exprs: Sequence[z3.ExprRef]) -> z3.ExprRef:
        """Lower Boolean Distinct into the supported basis."""
        if len(exprs) <= 1:
            return z3.BoolVal(True)
        if len(exprs) == 2:
            return self._lower_xor(exprs)
        return z3.BoolVal(False)

    def _encode_literal(
        self,
        expr: z3.ExprRef,
        side: ClausePartition,
        clauses: List[Clause],
        cache: Dict[str, int],
        atom_vars: Set[int],
    ) -> int:
        """Encode a lowered Boolean formula as a propositional literal."""
        if z3.is_true(expr):
            return self._constant_literal(True, side, clauses, cache)

        if z3.is_false(expr):
            return self._constant_literal(False, side, clauses, cache)

        if self._is_atomic_boolean(expr):
            var = self._get_atom_var(expr)
            atom_vars.add(var)
            return var

        kind = expr.decl().kind()
        if kind == z3.Z3_OP_NOT:
            return -self._encode_literal(expr.arg(0), side, clauses, cache, atom_vars)

        key = expr.sexpr()
        if key in cache:
            return cache[key]

        child_literals = [
            self._encode_literal(child, side, clauses, cache, atom_vars)
            for child in expr.children()
        ]
        var = self._new_aux_var(side)
        cache[key] = var

        if kind == z3.Z3_OP_AND:
            for literal in child_literals:
                self._append_clause(clauses, [-var, literal])
            self._append_clause(clauses, [var] + [-literal for literal in child_literals])
            return var

        if kind == z3.Z3_OP_OR:
            self._append_clause(clauses, [-var] + list(child_literals))
            for literal in child_literals:
                self._append_clause(clauses, [var, -literal])
            return var

        raise ValueError("formula was not lowered to the supported Boolean basis")

    def _constant_literal(
        self,
        value: bool,
        side: ClausePartition,
        clauses: List[Clause],
        cache: Dict[str, int],
    ) -> int:
        """Encode a Boolean constant as a unit-defined local variable."""
        key = "__const_true__" if value else "__const_false__"
        if key in cache:
            return cache[key]

        var = self._new_aux_var(side)
        cache[key] = var
        self._append_clause(clauses, [var] if value else [-var])
        return var

    def _append_clause(self, clauses: List[Clause], literals: Sequence[int]) -> None:
        """Append a non-tautological clause to the clause list."""
        clause = _normalize_clause(literals)
        if clause is not None:
            clauses.append(clause)

    def _is_atomic_boolean(self, expr: z3.ExprRef) -> bool:
        """Whether `expr` should be treated as a propositional atom."""
        kind = expr.decl().kind()
        return kind not in (z3.Z3_OP_NOT, z3.Z3_OP_AND, z3.Z3_OP_OR)

    def _get_atom_var(self, expr: z3.ExprRef) -> int:
        """Get or create a propositional variable for an original atom."""
        key = expr.sexpr()
        if key not in self._atom_to_var:
            var = self._next_var
            self._next_var += 1
            self._atom_to_var[key] = var
            self.var_to_expr[var] = expr
        return self._atom_to_var[key]

    def _new_aux_var(self, side: ClausePartition) -> int:
        """Allocate a fresh partition-local Tseitin variable."""
        var = self._next_var
        self._next_var += 1
        self.var_to_expr[var] = z3.Bool(f"__{side.value.lower()}_aux_{var}")
        return var


class ResolutionProofSystem:
    """Naive resolution saturation for unsatisfiable propositional clause sets."""

    def prove_unsat(
        self,
        clauses_a: Sequence[Sequence[int]],
        clauses_b: Sequence[Sequence[int]],
        var_to_expr: Dict[int, z3.ExprRef],
    ) -> ResolutionProof:
        """Build a resolution proof of contradiction if one exists."""
        steps: List[ResolutionStep] = []
        clause_to_step: Dict[Clause, int] = {}
        agenda: Deque[int] = deque()
        positive_index: Dict[int, Set[int]] = {}
        negative_index: Dict[int, Set[int]] = {}
        vars_a: Set[int] = set()
        vars_b: Set[int] = set()

        def index_clause(step_index: int, clause: Clause) -> None:
            for literal in clause:
                store = positive_index if literal > 0 else negative_index
                store.setdefault(abs(literal), set()).add(step_index)

        def add_step(step: ResolutionStep) -> int:
            if step.clause in clause_to_step:
                return clause_to_step[step.clause]

            index = len(steps)
            steps.append(step)
            clause_to_step[step.clause] = index
            agenda.append(index)
            index_clause(index, step.clause)
            return index

        for raw_clause in clauses_a:
            clause = _normalize_clause(raw_clause)
            if clause is None:
                continue
            vars_a.update(abs(literal) for literal in clause)
            root = add_step(
                ResolutionStep(clause=clause, partition=ClausePartition.A)
            )
            if not clause:
                return ResolutionProof(steps, root, var_to_expr, vars_a, vars_b)

        for raw_clause in clauses_b:
            clause = _normalize_clause(raw_clause)
            if clause is None:
                continue
            vars_b.update(abs(literal) for literal in clause)
            root = add_step(
                ResolutionStep(clause=clause, partition=ClausePartition.B)
            )
            if not clause:
                return ResolutionProof(steps, root, var_to_expr, vars_a, vars_b)

        while agenda:
            current = agenda.popleft()
            current_clause = steps[current].clause

            for literal in current_clause:
                opposite_index = (
                    negative_index.get(abs(literal), set())
                    if literal > 0
                    else positive_index.get(abs(literal), set())
                )
                for other in list(opposite_index):
                    if other >= current:
                        continue

                    resolvent = _resolve_clauses(
                        current_clause, steps[other].clause, abs(literal)
                    )
                    if resolvent is None:
                        continue

                    root = add_step(
                        ResolutionStep(
                            clause=resolvent,
                            left=other,
                            right=current,
                            pivot=abs(literal),
                        )
                    )
                    if not resolvent:
                        return ResolutionProof(steps, root, var_to_expr, vars_a, vars_b)
        raise ValueError(
            "A and B are satisfiable together; no resolution refutation exists"
        )


class BooleanInterpolant:
    """Facade for proof-based Boolean interpolation."""

    @staticmethod
    def build_proof(fml_a: z3.ExprRef, fml_b: z3.ExprRef) -> ResolutionProof:
        """Clausify both inputs and produce a resolution refutation."""
        clausifier = PropositionalClausifier()
        encoding_a = clausifier.clausify(fml_a, ClausePartition.A)
        encoding_b = clausifier.clausify(fml_b, ClausePartition.B)
        proof_system = ResolutionProofSystem()
        proof = proof_system.prove_unsat(
            encoding_a.clauses, encoding_b.clauses, clausifier.var_to_expr
        )
        proof.vars_a = encoding_a.vars_in_formula
        proof.vars_b = encoding_b.vars_in_formula
        return proof

    @staticmethod
    def compute_itp(fml_a: z3.ExprRef, fml_b: z3.ExprRef) -> z3.ExprRef:
        """Compute a Craig interpolant for two unsatisfiable Boolean formulas."""
        proof = BooleanInterpolant.build_proof(fml_a, fml_b)
        return proof.extract_interpolant()

    @staticmethod
    def build_proof_from_cnf(
        fml_a: CNF,
        fml_b: CNF,
        var_to_expr: Optional[Dict[int, z3.ExprRef]] = None,
    ) -> ResolutionProof:
        """Build a proof directly from CNF clause sets."""
        all_vars = set()
        vars_a = set()
        vars_b = set()

        for clause in fml_a.clauses:
            for literal in clause:
                all_vars.add(abs(literal))
                vars_a.add(abs(literal))

        for clause in fml_b.clauses:
            for literal in clause:
                all_vars.add(abs(literal))
                vars_b.add(abs(literal))

        expr_map: Dict[int, z3.ExprRef] = {}
        if var_to_expr is not None:
            expr_map.update(var_to_expr)
        for var in all_vars:
            expr_map.setdefault(var, z3.Bool(f"v{var}"))

        proof_system = ResolutionProofSystem()
        proof = proof_system.prove_unsat(fml_a.clauses, fml_b.clauses, expr_map)
        proof.vars_a = vars_a
        proof.vars_b = vars_b
        return proof

    @staticmethod
    def compute_itp_from_cnf(
        fml_a: CNF,
        fml_b: CNF,
        var_to_expr: Optional[Dict[int, z3.ExprRef]] = None,
    ) -> z3.ExprRef:
        """Compute an interpolant directly from two CNF formulas."""
        proof = BooleanInterpolant.build_proof_from_cnf(fml_a, fml_b, var_to_expr)
        return proof.extract_interpolant()


def demo_itp() -> None:
    """Small demo for the proof-based interpolant engine."""
    p, q = z3.Bools("p q")
    fml_a = z3.And(z3.Or(p, q), z3.Not(p))
    fml_b = z3.Not(q)

    proof = BooleanInterpolant.build_proof(fml_a, fml_b)
    interpolant = proof.extract_interpolant()

    print("Proof valid:", proof.validate())
    print("Interpolant:", interpolant)


if __name__ == "__main__":
    demo_itp()
