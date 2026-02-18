"""Bounded second-order decision procedure (productionized PoC).

This module decides closed formulas in a finite two-sorted language:
- naturals in a bounded universe ``{0, ..., universe_size-1}``
- sets as subsets of that finite universe

It is inspired by reverse-math style syntax (number variables + set variables),
but it is not a full implementation of ``RCA_0``/``ACA_0``.
"""

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union


NatEnv = Dict[str, int]
SetEnv = Dict[str, FrozenSet[int]]
SexpNode = Union[str, int, List["SexpNode"]]


class DecisionError(Exception):
    """Base class for decision-procedure errors."""


class UnboundVariableError(DecisionError):
    """Raised when evaluation encounters an unbound variable."""


class InvalidFormulaError(DecisionError):
    """Raised when the input formula is malformed for this fragment."""


class NatTerm:
    """Base class for numeric terms in the finite arithmetic fragment."""


@dataclass(frozen=True)
class NatConst(NatTerm):
    """Natural number constant."""

    value: int


@dataclass(frozen=True)
class NatVar(NatTerm):
    """Natural number variable."""

    name: str


@dataclass(frozen=True)
class NatAdd(NatTerm):
    """Addition term for small arithmetic expressions."""

    left: NatTerm
    right: NatTerm


class Formula:
    """Base class for formulas in the bounded two-sorted language."""


@dataclass(frozen=True)
class BoolConst(Formula):
    """Boolean literal formula."""

    value: bool


@dataclass(frozen=True)
class NatEq(Formula):
    """Equality between numeric terms."""

    left: NatTerm
    right: NatTerm


@dataclass(frozen=True)
class NatLt(Formula):
    """Strict comparison between numeric terms."""

    left: NatTerm
    right: NatTerm


@dataclass(frozen=True)
class NatIn(Formula):
    """Set membership atom: ``element in set_var``."""

    element: NatTerm
    set_var: str


@dataclass(frozen=True)
class Not(Formula):
    """Logical negation."""

    body: Formula


@dataclass(frozen=True)
class And(Formula):
    """Logical conjunction over multiple formulas."""

    parts: Sequence[Formula]


@dataclass(frozen=True)
class Or(Formula):
    """Logical disjunction over multiple formulas."""

    parts: Sequence[Formula]


@dataclass(frozen=True)
class Implies(Formula):
    """Logical implication."""

    left: Formula
    right: Formula


@dataclass(frozen=True)
class ForallNat(Formula):
    """Bounded first-order universal quantification over naturals."""

    var: str
    bound: int
    body: Formula


@dataclass(frozen=True)
class ExistsNat(Formula):
    """Bounded first-order existential quantification over naturals."""

    var: str
    bound: int
    body: Formula


@dataclass(frozen=True)
class ForallSet(Formula):
    """Second-order universal quantification over finite subsets."""

    var: str
    body: Formula


@dataclass(frozen=True)
class ExistsSet(Formula):
    """Second-order existential quantification over finite subsets."""

    var: str
    body: Formula


@dataclass(frozen=True)
class DecisionResult:
    """Decision outcome for a closed bounded formula."""

    valid: bool
    evaluated_states: int
    evaluated_set_assignments: int


@dataclass
class _EvalStats:
    """Internal counters used for transparency and basic profiling."""

    evaluated_states: int = 0
    evaluated_set_assignments: int = 0


_SENTINEL = object()


class FiniteSecondOrderDecisionProcedure:
    """Exact decision procedure for a finite two-sorted fragment.

    Args:
        universe_size: Size of the finite natural universe ``{0, ..., n-1}``.
        max_universe_size: Safety limit to avoid accidental exponential blow-ups.
            Set to ``None`` to disable.

    Raises:
        ValueError: If sizes are invalid.
    """

    def __init__(
        self, universe_size: int, max_universe_size: Optional[int] = 12
    ) -> None:
        if universe_size < 0:
            raise ValueError("universe_size must be non-negative")
        if max_universe_size is not None and max_universe_size < 0:
            raise ValueError("max_universe_size must be non-negative or None")
        if max_universe_size is not None and universe_size > max_universe_size:
            raise ValueError(
                "universe_size={} exceeds max_universe_size={}".format(
                    universe_size, max_universe_size
                )
            )

        self.universe_size = universe_size
        self._universe_values: Tuple[int, ...] = tuple(range(universe_size))
        self._all_subsets_cache: Tuple[FrozenSet[int], ...] = self._build_all_subsets()

    def decide(self, formula: Formula) -> DecisionResult:
        """Decide a closed formula in finite semantics.

        Raises:
            InvalidFormulaError: If the formula is not closed or malformed.
            UnboundVariableError: If evaluation finds a missing binding.
            TypeError: If unsupported AST nodes are used.
        """
        self._validate_closed_formula(formula)

        stats = _EvalStats()
        valid = self._eval_formula(
            formula=formula,
            nat_env={},
            set_env={},
            stats=stats,
        )
        return DecisionResult(
            valid=valid,
            evaluated_states=stats.evaluated_states,
            evaluated_set_assignments=stats.evaluated_set_assignments,
        )

    def _build_all_subsets(self) -> Tuple[FrozenSet[int], ...]:
        subsets: List[FrozenSet[int]] = []
        universe = self._universe_values
        for size in range(len(universe) + 1):
            for tuple_subset in combinations(universe, size):
                subsets.append(frozenset(tuple_subset))
        return tuple(subsets)

    def _eval_term(self, term: NatTerm, nat_env: NatEnv) -> int:
        if isinstance(term, NatConst):
            return term.value

        if isinstance(term, NatVar):
            if term.name not in nat_env:
                raise UnboundVariableError(
                    "Unbound nat variable: {}".format(term.name)
                )
            return nat_env[term.name]

        if isinstance(term, NatAdd):
            return self._eval_term(term.left, nat_env) + self._eval_term(
                term.right, nat_env
            )

        raise TypeError("Unsupported term type: {}".format(type(term).__name__))

    def _eval_formula(
        self,
        formula: Formula,
        nat_env: NatEnv,
        set_env: SetEnv,
        stats: _EvalStats,
    ) -> bool:
        stats.evaluated_states += 1

        if isinstance(formula, BoolConst):
            return formula.value

        if isinstance(formula, NatEq):
            return self._eval_term(formula.left, nat_env) == self._eval_term(
                formula.right, nat_env
            )

        if isinstance(formula, NatLt):
            return self._eval_term(formula.left, nat_env) < self._eval_term(
                formula.right, nat_env
            )

        if isinstance(formula, NatIn):
            if formula.set_var not in set_env:
                raise UnboundVariableError(
                    "Unbound set variable: {}".format(formula.set_var)
                )
            return self._eval_term(formula.element, nat_env) in set_env[
                formula.set_var
            ]

        if isinstance(formula, Not):
            return not self._eval_formula(formula.body, nat_env, set_env, stats)

        if isinstance(formula, And):
            return all(
                self._eval_formula(part, nat_env, set_env, stats)
                for part in formula.parts
            )

        if isinstance(formula, Or):
            return any(
                self._eval_formula(part, nat_env, set_env, stats)
                for part in formula.parts
            )

        if isinstance(formula, Implies):
            return (not self._eval_formula(formula.left, nat_env, set_env, stats)) or (
                self._eval_formula(formula.right, nat_env, set_env, stats)
            )

        if isinstance(formula, ForallNat):
            self._validate_bound(formula.bound)
            upper_bound = min(formula.bound, self.universe_size)
            old_value = nat_env.get(formula.var, _SENTINEL)
            try:
                for value in range(upper_bound):
                    nat_env[formula.var] = value
                    if not self._eval_formula(formula.body, nat_env, set_env, stats):
                        return False
                return True
            finally:
                if old_value is _SENTINEL:
                    nat_env.pop(formula.var, None)
                else:
                    nat_env[formula.var] = old_value

        if isinstance(formula, ExistsNat):
            self._validate_bound(formula.bound)
            upper_bound = min(formula.bound, self.universe_size)
            old_value = nat_env.get(formula.var, _SENTINEL)
            try:
                for value in range(upper_bound):
                    nat_env[formula.var] = value
                    if self._eval_formula(formula.body, nat_env, set_env, stats):
                        return True
                return False
            finally:
                if old_value is _SENTINEL:
                    nat_env.pop(formula.var, None)
                else:
                    nat_env[formula.var] = old_value

        if isinstance(formula, ForallSet):
            old_value = set_env.get(formula.var, _SENTINEL)
            try:
                for subset in self._all_subsets_cache:
                    stats.evaluated_set_assignments += 1
                    set_env[formula.var] = subset
                    if not self._eval_formula(formula.body, nat_env, set_env, stats):
                        return False
                return True
            finally:
                if old_value is _SENTINEL:
                    set_env.pop(formula.var, None)
                else:
                    set_env[formula.var] = old_value

        if isinstance(formula, ExistsSet):
            old_value = set_env.get(formula.var, _SENTINEL)
            try:
                for subset in self._all_subsets_cache:
                    stats.evaluated_set_assignments += 1
                    set_env[formula.var] = subset
                    if self._eval_formula(formula.body, nat_env, set_env, stats):
                        return True
                return False
            finally:
                if old_value is _SENTINEL:
                    set_env.pop(formula.var, None)
                else:
                    set_env[formula.var] = old_value

        raise TypeError("Unsupported formula type: {}".format(type(formula).__name__))

    def _validate_bound(self, bound: int) -> None:
        if not isinstance(bound, int):
            raise InvalidFormulaError("Nat quantifier bound must be an int")
        if bound < 0:
            raise InvalidFormulaError("Nat quantifier bound must be non-negative")

    def _validate_closed_formula(self, formula: Formula) -> None:
        free_nat, free_set = free_variables(formula)
        if free_nat or free_set:
            raise InvalidFormulaError(
                "Formula must be closed. Free nat vars: {}; free set vars: {}".format(
                    sorted(free_nat), sorted(free_set)
                )
            )


def free_variables(formula: Formula) -> Tuple[Set[str], Set[str]]:
    """Return ``(free_nat_vars, free_set_vars)`` for a formula."""

    def term_free_vars(term: NatTerm) -> Set[str]:
        if isinstance(term, NatConst):
            return set()
        if isinstance(term, NatVar):
            return {term.name}
        if isinstance(term, NatAdd):
            return term_free_vars(term.left) | term_free_vars(term.right)
        raise TypeError("Unsupported term type: {}".format(type(term).__name__))

    def visit(
        fml: Formula,
        bound_nat: Set[str],
        bound_set: Set[str],
    ) -> Tuple[Set[str], Set[str]]:
        if isinstance(fml, BoolConst):
            return set(), set()

        if isinstance(fml, NatEq) or isinstance(fml, NatLt):
            nat_vars = term_free_vars(fml.left) | term_free_vars(fml.right)
            return nat_vars - bound_nat, set()

        if isinstance(fml, NatIn):
            nat_vars = term_free_vars(fml.element) - bound_nat
            set_vars = set()
            if fml.set_var not in bound_set:
                set_vars.add(fml.set_var)
            return nat_vars, set_vars

        if isinstance(fml, Not):
            return visit(fml.body, bound_nat, bound_set)

        if isinstance(fml, And) or isinstance(fml, Or):
            nat_acc: Set[str] = set()
            set_acc: Set[str] = set()
            for part in fml.parts:
                n, s = visit(part, bound_nat, bound_set)
                nat_acc |= n
                set_acc |= s
            return nat_acc, set_acc

        if isinstance(fml, Implies):
            left_nat, left_set = visit(fml.left, bound_nat, bound_set)
            right_nat, right_set = visit(fml.right, bound_nat, bound_set)
            return left_nat | right_nat, left_set | right_set

        if isinstance(fml, ForallNat) or isinstance(fml, ExistsNat):
            if not isinstance(fml.var, str) or not fml.var:
                raise InvalidFormulaError("Nat quantifier variable name must be non-empty")
            return visit(fml.body, bound_nat | {fml.var}, bound_set)

        if isinstance(fml, ForallSet) or isinstance(fml, ExistsSet):
            if not isinstance(fml.var, str) or not fml.var:
                raise InvalidFormulaError("Set quantifier variable name must be non-empty")
            return visit(fml.body, bound_nat, bound_set | {fml.var})

        raise TypeError("Unsupported formula type: {}".format(type(fml).__name__))

    return visit(formula, set(), set())


def iff(left: Formula, right: Formula) -> Formula:
    """Build logical equivalence using implication/conjunction."""
    return And([Implies(left, right), Implies(right, left)])


def parse_formula_text(source: str) -> Formula:
    """Parse a formula from a compact S-expression DSL.

    Supported forms:
    - ``true`` / ``false``
    - ``(eq t1 t2)``, ``(lt t1 t2)``, ``(in t X)``
    - ``(not f)``, ``(and f1 f2 ...)``, ``(or f1 f2 ...)``
    - ``(implies f1 f2)``, ``(iff f1 f2)``
    - ``(forall_nat n bound f)``, ``(exists_nat n bound f)``
    - ``(forall_set X f)``, ``(exists_set X f)``
    """
    node = _parse_single_sexp(source)
    return _formula_from_sexp(node)


def parse_term_text(source: str) -> NatTerm:
    """Parse a nat term from the same S-expression DSL.

    Supported terms:
    - integer literals, e.g. ``0``, ``42``
    - identifiers, e.g. ``n``
    - ``(+ t1 t2)``
    """
    node = _parse_single_sexp(source)
    return _term_from_sexp(node)


def _parse_single_sexp(source: str) -> SexpNode:
    tokens = _tokenize_sexp(source)
    if not tokens:
        raise InvalidFormulaError("Expected a non-empty S-expression")

    node, next_index = _read_sexp(tokens, 0)
    if next_index != len(tokens):
        raise InvalidFormulaError("Unexpected trailing tokens")
    return node


def _tokenize_sexp(source: str) -> List[str]:
    source = source.strip()
    token_pattern = re.compile(r"\s*([()]|[^\s()]+)")
    tokens: List[str] = []
    position = 0
    while position < len(source):
        match = token_pattern.match(source, position)
        if match is None:
            raise InvalidFormulaError(
                "Could not tokenize input near index {}".format(position)
            )
        token = match.group(1)
        tokens.append(token)
        position = match.end()
    return tokens


def _read_sexp(tokens: Sequence[str], start_index: int) -> Tuple[SexpNode, int]:
    if start_index >= len(tokens):
        raise InvalidFormulaError("Unexpected end of input")

    token = tokens[start_index]
    if token == "(":
        values: List[SexpNode] = []
        index = start_index + 1
        while True:
            if index >= len(tokens):
                raise InvalidFormulaError("Unbalanced parentheses")
            if tokens[index] == ")":
                return values, index + 1
            child, index = _read_sexp(tokens, index)
            values.append(child)

    if token == ")":
        raise InvalidFormulaError("Unexpected ')'")

    if token.lstrip("-").isdigit():
        return int(token), start_index + 1
    return token, start_index + 1


def _formula_from_sexp(node: SexpNode) -> Formula:
    if isinstance(node, str):
        if node == "true":
            return BoolConst(True)
        if node == "false":
            return BoolConst(False)
        raise InvalidFormulaError(
            "Unexpected atom in formula position: '{}'".format(node)
        )

    if isinstance(node, int):
        raise InvalidFormulaError("Unexpected integer in formula position")

    if not node:
        raise InvalidFormulaError("Empty S-expression is not a formula")
    if not isinstance(node[0], str):
        raise InvalidFormulaError("Formula operator must be a symbol")

    op = node[0]
    args = node[1:]

    if op == "eq":
        _expect_arity(op, args, 2)
        return NatEq(_term_from_sexp(args[0]), _term_from_sexp(args[1]))

    if op == "lt":
        _expect_arity(op, args, 2)
        return NatLt(_term_from_sexp(args[0]), _term_from_sexp(args[1]))

    if op == "in":
        _expect_arity(op, args, 2)
        if not isinstance(args[1], str):
            raise InvalidFormulaError("'in' expects a set variable symbol")
        return NatIn(_term_from_sexp(args[0]), args[1])

    if op == "not":
        _expect_arity(op, args, 1)
        return Not(_formula_from_sexp(args[0]))

    if op == "and":
        if len(args) < 2:
            raise InvalidFormulaError("'and' expects at least 2 operands")
        return And([_formula_from_sexp(arg) for arg in args])

    if op == "or":
        if len(args) < 2:
            raise InvalidFormulaError("'or' expects at least 2 operands")
        return Or([_formula_from_sexp(arg) for arg in args])

    if op == "implies":
        _expect_arity(op, args, 2)
        return Implies(_formula_from_sexp(args[0]), _formula_from_sexp(args[1]))

    if op == "iff":
        _expect_arity(op, args, 2)
        return iff(_formula_from_sexp(args[0]), _formula_from_sexp(args[1]))

    if op == "forall_nat":
        _expect_arity(op, args, 3)
        if not isinstance(args[0], str):
            raise InvalidFormulaError("'forall_nat' expects a variable name")
        if not isinstance(args[1], int):
            raise InvalidFormulaError("'forall_nat' expects an integer bound")
        return ForallNat(args[0], args[1], _formula_from_sexp(args[2]))

    if op == "exists_nat":
        _expect_arity(op, args, 3)
        if not isinstance(args[0], str):
            raise InvalidFormulaError("'exists_nat' expects a variable name")
        if not isinstance(args[1], int):
            raise InvalidFormulaError("'exists_nat' expects an integer bound")
        return ExistsNat(args[0], args[1], _formula_from_sexp(args[2]))

    if op == "forall_set":
        _expect_arity(op, args, 2)
        if not isinstance(args[0], str):
            raise InvalidFormulaError("'forall_set' expects a variable name")
        return ForallSet(args[0], _formula_from_sexp(args[1]))

    if op == "exists_set":
        _expect_arity(op, args, 2)
        if not isinstance(args[0], str):
            raise InvalidFormulaError("'exists_set' expects a variable name")
        return ExistsSet(args[0], _formula_from_sexp(args[1]))

    raise InvalidFormulaError("Unknown formula operator: '{}'".format(op))


def _term_from_sexp(node: SexpNode) -> NatTerm:
    if isinstance(node, int):
        return NatConst(node)

    if isinstance(node, str):
        return NatVar(node)

    if not node:
        raise InvalidFormulaError("Empty S-expression is not a term")
    if not isinstance(node[0], str):
        raise InvalidFormulaError("Term operator must be a symbol")

    op = node[0]
    args = node[1:]
    if op == "+":
        _expect_arity(op, args, 2)
        return NatAdd(_term_from_sexp(args[0]), _term_from_sexp(args[1]))

    raise InvalidFormulaError("Unknown term operator: '{}'".format(op))


def _expect_arity(op: str, args: Sequence[SexpNode], arity: int) -> None:
    if len(args) != arity:
        raise InvalidFormulaError(
            "Operator '{}' expects {} args, got {}".format(op, arity, len(args))
        )


__all__ = [
    "And",
    "BoolConst",
    "DecisionError",
    "DecisionResult",
    "ExistsNat",
    "ExistsSet",
    "FiniteSecondOrderDecisionProcedure",
    "ForallNat",
    "ForallSet",
    "Formula",
    "Implies",
    "InvalidFormulaError",
    "NatAdd",
    "NatConst",
    "NatEq",
    "NatIn",
    "NatLt",
    "NatTerm",
    "NatVar",
    "Not",
    "Or",
    "UnboundVariableError",
    "free_variables",
    "iff",
    "parse_formula_text",
    "parse_term_text",
]
