"""Semantic minimization algorithms for three-valued propositional logic.

The implementations in this module follow the constructions from Reps,
Loginov, and Sagiv, "Semantic Minimization of 3-Valued Propositional
Formulae" (LICS 2002).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union, cast

from aria.utils.bdd import BDDNode, OBDD
from aria.utils.bdd.BDD import BDDNonTerminalNode
from aria.utils.bdd.ordering import ListOrdering

from .propositional import (
    STRONG_KLEENE,
    And,
    BinaryFormula,
    Constant,
    Formula,
    Iff,
    Implies,
    NAryFormula,
    Nand,
    NaryAnd,
    NaryOr,
    Not,
    Nor,
    Or,
    Semantics,
    TruthLike,
    TruthValue,
    UnaryFormula,
    Variable,
    Xor,
    all_valuations,
    classical_valuations,
    conjoin,
    disjoin,
    evaluate,
)

BooleanAssignment = tuple[tuple[str, bool], ...]
PartialBooleanAssignment = tuple[tuple[str, Optional[bool]], ...]
ThreeValuedAssignment = tuple[tuple[str, TruthValue], ...]
BooleanFunctionInput = Union[Mapping[BooleanAssignment, bool], OBDD]


@dataclass(frozen=True)
class InformationJoinFormula(Formula):
    left: Formula
    right: Formula

    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        left_value = self.left.evaluate(valuation, semantics)
        right_value = self.right.evaluate(valuation, semantics)
        if left_value is right_value:
            return left_value
        if left_value is TruthValue.UNKNOWN:
            return right_value
        if right_value is TruthValue.UNKNOWN:
            return left_value
        return TruthValue.UNKNOWN

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()

    def subformulas(self) -> tuple[Formula, ...]:
        return (self,) + self.left.subformulas() + self.right.subformulas()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return InformationJoinFormula(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )

    def to_string(self, unicode: bool = False) -> str:
        symbol = "⊔" if unicode else "join"
        left = self.left.to_string(unicode=unicode)
        right = self.right.to_string(unicode=unicode)
        return f"({left} {symbol} {right})"


@dataclass(frozen=True)
class BooleanPairOBDD:
    """BDD representation of a three-valued function on definite inputs.

    ``is_true`` is the Boolean characteristic function of truth value ``TRUE``.
    ``is_nonfalse`` is the Boolean characteristic function of values that are
    not ``FALSE`` (i.e. ``TRUE`` or ``UNKNOWN``).
    """

    is_true: OBDD
    is_nonfalse: OBDD

    @property
    def ordering(self) -> ListOrdering:
        if not isinstance(self.is_true.ordering, ListOrdering):
            raise TypeError("Expected list-based OBDD ordering")
        return cast(ListOrdering, self.is_true.ordering)


def _normalize_boolean_assignment(
    valuation: Mapping[str, bool],
    variables: Sequence[str],
) -> BooleanAssignment:
    return tuple((name, bool(valuation[name])) for name in variables)


def _normalize_partial_boolean_assignment(
    valuation: Mapping[str, Optional[bool]],
    variables: Sequence[str],
) -> PartialBooleanAssignment:
    return tuple((name, valuation.get(name)) for name in variables)


def _normalize_threevalued_assignment(
    valuation: Mapping[str, TruthLike],
    variables: Sequence[str],
) -> ThreeValuedAssignment:
    return tuple((name, TruthValue.coerce(valuation[name])) for name in variables)


def _partial_to_formula(cube: PartialBooleanAssignment) -> Formula:
    literals = []
    for name, value in cube:
        if value is None:
            continue
        atom = Variable(name)
        literals.append(atom if value else Not(atom))
    return conjoin(*literals)


def _cube_sort_key(cube: PartialBooleanAssignment) -> tuple[tuple[str, int], ...]:
    return tuple((name, 2 if value is None else int(value)) for name, value in cube)


def _dual_clause_to_formula(cube: PartialBooleanAssignment) -> Formula:
    literals = []
    for name, value in cube:
        if value is None:
            continue
        atom = Variable(name)
        literals.append(Not(atom) if value else atom)
    return disjoin(*literals)


def _all_partial_boolean_assignments(
    variables: Sequence[str],
) -> Iterable[PartialBooleanAssignment]:
    options = (False, True, None)
    for values in product(options, repeat=len(variables)):
        yield tuple(zip(variables, values))


def _strictly_less_definite(
    less_definite: ThreeValuedAssignment,
    base: ThreeValuedAssignment,
) -> bool:
    candidate_map = dict(less_definite)
    base_map = dict(base)
    changed = False
    for name, base_value in base_map.items():
        candidate_value = candidate_map[name]
        if base_value is TruthValue.UNKNOWN:
            if candidate_value is not TruthValue.UNKNOWN:
                return False
            continue
        if candidate_value is TruthValue.UNKNOWN:
            changed = True
            continue
        if candidate_value is not base_value:
            return False
    return changed


def _cube_subsumes(
    left: PartialBooleanAssignment,
    right: PartialBooleanAssignment,
) -> bool:
    right_map = dict(right)
    return all(value is None or right_map.get(name) is value for name, value in left)


def _normalize_cube(
    cube: Iterable[tuple[str, Optional[bool]]],
    variables: Sequence[str],
) -> PartialBooleanAssignment:
    cube_map = dict(cube)
    return tuple((name, cube_map.get(name)) for name in variables)


def _restrict_obdd(obdd: OBDD, cube: PartialBooleanAssignment) -> OBDD:
    restricted = obdd
    for name, value in cube:
        if value is None:
            continue
        restricted = restricted.restrict(name, value)
        if restricted == 0 or restricted == 1:
            return restricted
    return restricted


def _enumerate_true_paths(node: BDDNode) -> list[tuple[tuple[str, bool], ...]]:
    if getattr(node, "value", None) is True:
        return [tuple()]
    if getattr(node, "value", None) is False:
        return []

    if not isinstance(node, BDDNonTerminalNode):
        return []
    node = cast(BDDNonTerminalNode, node)

    paths = []
    for suffix in _enumerate_true_paths(node.low):
        paths.append(((node.var, False),) + suffix)
    for suffix in _enumerate_true_paths(node.high):
        paths.append(((node.var, True),) + suffix)
    return paths


def _reduce_implicant_cube(
    obdd: OBDD,
    cube: PartialBooleanAssignment,
) -> PartialBooleanAssignment:
    reduced = list(cube)
    changed = True
    while changed:
        changed = False
        for index, (name, value) in enumerate(list(reduced)):
            if value is None:
                continue
            trial = list(reduced)
            trial[index] = (name, None)
            if _restrict_obdd(obdd, tuple(trial)) == 1:
                reduced = trial
                changed = True
                break
    return tuple(reduced)


def build_obdd_from_boolean_function(
    variables: Sequence[str],
    evaluator: Callable[[Mapping[str, bool]], bool],
) -> OBDD:
    """Build an OBDD for a total Boolean function over ``variables``."""

    ordered_variables = list(variables)
    cache = {}

    def build(index: int, prefix: tuple[tuple[str, bool], ...]) -> BDDNode:
        key = (index, prefix)
        if key in cache:
            return cache[key]
        if index >= len(ordered_variables):
            cache[key] = BDDNode(bool(evaluator(dict(prefix))))
            return cache[key]

        name = ordered_variables[index]
        low = build(index + 1, prefix + ((name, False),))
        high = build(index + 1, prefix + ((name, True),))
        cache[key] = BDDNode(name, low, high)
        return cache[key]

    return OBDD(build(0, tuple()), ordered_variables, check_ordering=False)


def _constant_obdd(value: bool, variables: Sequence[str]) -> OBDD:
    return OBDD(BDDNode(value), list(variables), check_ordering=False)


def _variable_obdd(name: str, variables: Sequence[str]) -> OBDD:
    return OBDD(
        BDDNode(name, BDDNode(False), BDDNode(True)),
        list(variables),
        check_ordering=False,
    )


def _pair_not(pair: BooleanPairOBDD) -> BooleanPairOBDD:
    return BooleanPairOBDD(is_true=~pair.is_nonfalse, is_nonfalse=~pair.is_true)


def _pair_and(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return BooleanPairOBDD(
        is_true=left.is_true & right.is_true,
        is_nonfalse=left.is_nonfalse & right.is_nonfalse,
    )


def _pair_or(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return BooleanPairOBDD(
        is_true=left.is_true | right.is_true,
        is_nonfalse=left.is_nonfalse | right.is_nonfalse,
    )


def _pair_join(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return BooleanPairOBDD(
        is_true=(left.is_true & right.is_nonfalse)
        | (right.is_true & left.is_nonfalse),
        is_nonfalse=left.is_nonfalse | right.is_nonfalse,
    )


def _pair_implies(
    left: BooleanPairOBDD, right: BooleanPairOBDD
) -> BooleanPairOBDD:
    return _pair_or(_pair_not(left), right)


def _pair_iff(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return _pair_and(_pair_implies(left, right), _pair_implies(right, left))


def _pair_xor(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return _pair_not(_pair_iff(left, right))


def _pair_nand(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return _pair_not(_pair_and(left, right))


def _pair_nor(left: BooleanPairOBDD, right: BooleanPairOBDD) -> BooleanPairOBDD:
    return _pair_not(_pair_or(left, right))


def _fold_pair(
    operands: Sequence[BooleanPairOBDD],
    neutral: BooleanPairOBDD,
    operator: Callable[[BooleanPairOBDD, BooleanPairOBDD], BooleanPairOBDD],
) -> BooleanPairOBDD:
    result = neutral
    for operand in operands:
        result = operator(result, operand)
    return result


def direct_supervaluation_obdd_pair(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> BooleanPairOBDD:
    """Build the Section 5 Boolean-pair representation directly from syntax.

    For strong Kleene semantics, this follows the paper faithfully by
    traversing the formula bottom-up and constructing the Boolean pair using
    BDD operations rather than enumerating all Boolean assignments.

    For other semantics, the function falls back to the oracle-based builder.
    """

    if semantics != STRONG_KLEENE:
        true_obdd, nonfalse_obdd = supervaluation_obdd_pair(formula, semantics)
        return BooleanPairOBDD(true_obdd, nonfalse_obdd)

    variables = sorted(formula.variables())
    pair_true = BooleanPairOBDD(
        _constant_obdd(True, variables), _constant_obdd(True, variables)
    )
    pair_false = BooleanPairOBDD(
        _constant_obdd(False, variables), _constant_obdd(False, variables)
    )
    pair_unknown = BooleanPairOBDD(
        _constant_obdd(False, variables), _constant_obdd(True, variables)
    )

    def translate(node: Formula) -> BooleanPairOBDD:
        if isinstance(node, Constant):
            if node.value is TruthValue.TRUE:
                return pair_true
            if node.value is TruthValue.FALSE:
                return pair_false
            return pair_unknown
        if isinstance(node, Variable):
            atom = _variable_obdd(node.name, variables)
            return BooleanPairOBDD(atom, atom)
        if isinstance(node, InformationJoinFormula):
            join_node = cast(InformationJoinFormula, node)
            return _pair_join(translate(join_node.left), translate(join_node.right))
        if isinstance(node, Not):
            unary_node = cast(Not, node)
            return _pair_not(translate(unary_node.operand))
        if isinstance(node, And):
            binary_node = cast(And, node)
            return _pair_and(translate(binary_node.left), translate(binary_node.right))
        if isinstance(node, Or):
            binary_node = cast(Or, node)
            return _pair_or(translate(binary_node.left), translate(binary_node.right))
        if isinstance(node, Implies):
            binary_node = cast(Implies, node)
            return _pair_implies(
                translate(binary_node.left), translate(binary_node.right)
            )
        if isinstance(node, Iff):
            binary_node = cast(Iff, node)
            return _pair_iff(translate(binary_node.left), translate(binary_node.right))
        if isinstance(node, Xor):
            binary_node = cast(Xor, node)
            return _pair_xor(translate(binary_node.left), translate(binary_node.right))
        if isinstance(node, Nand):
            binary_node = cast(Nand, node)
            return _pair_nand(
                translate(binary_node.left), translate(binary_node.right)
            )
        if isinstance(node, Nor):
            binary_node = cast(Nor, node)
            return _pair_nor(translate(binary_node.left), translate(binary_node.right))
        if isinstance(node, NaryAnd):
            nary_node = cast(NaryAnd, node)
            return _fold_pair(
                [translate(operand) for operand in nary_node.operands],
                pair_true,
                _pair_and,
            )
        if isinstance(node, NaryOr):
            nary_node = cast(NaryOr, node)
            return _fold_pair(
                [translate(operand) for operand in nary_node.operands],
                pair_false,
                _pair_or,
            )
        if isinstance(node, UnaryFormula):
            raise TypeError(f"Unsupported unary node: {node!r}")
        if isinstance(node, BinaryFormula):
            raise TypeError(f"Unsupported binary node: {node!r}")
        if isinstance(node, NAryFormula):
            raise TypeError(f"Unsupported n-ary node: {node!r}")
        raise TypeError(f"Unsupported formula node: {node!r}")

    return translate(formula)


def _as_obdd(
    function_values: BooleanFunctionInput,
    variables: Optional[Sequence[str]] = None,
) -> OBDD:
    if isinstance(function_values, OBDD):
        return function_values
    if not function_values:
        if variables is None:
            variables = []
        return OBDD(BDDNode(False), list(variables), check_ordering=False)

    active_variables = list(
        variables or [name for name, _ in next(iter(function_values.keys()))]
    )

    def evaluator(valuation: Mapping[str, bool]) -> bool:
        key = _normalize_boolean_assignment(valuation, active_variables)
        return bool(function_values[key])

    return build_obdd_from_boolean_function(active_variables, evaluator)


def information_join(left: Formula, right: Formula) -> Formula:
    """Return a formula implementing information-order join.

    The connective is definable in strong Kleene logic as
    ``(left & right) | (unknown & (left | right))``.
    """

    return InformationJoinFormula(left, right)


def supervaluation(
    formula: Formula,
    valuation: Mapping[str, TruthLike],
    semantics: Semantics = STRONG_KLEENE,
) -> TruthValue:
    """Evaluate the supervaluational semantics of a formula.

    The given valuation may be partial or three-valued. All classical
    completions represented by the valuation are evaluated, and their results
    are combined with information-order join.
    """

    variables = sorted(formula.variables())
    normalized = {
        name: TruthValue.coerce(valuation.get(name, TruthValue.UNKNOWN))
        for name in variables
    }
    free_variables = [
        name for name in variables if normalized[name] is TruthValue.UNKNOWN
    ]

    result: Optional[TruthValue] = None
    for completion in classical_valuations(free_variables):
        refined = dict(normalized)
        refined.update(completion)
        current = evaluate(formula, refined, semantics)
        if result is None:
            result = current
        else:
            if result is current:
                continue
            if result is TruthValue.UNKNOWN:
                result = current
            elif current is not TruthValue.UNKNOWN:
                result = TruthValue.UNKNOWN

    if result is None:
        return evaluate(formula, normalized, semantics)
    return result


def supervaluation_truth_table(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> list[tuple[dict[str, TruthValue], TruthValue]]:
    """Return the full supervaluational truth table of ``formula``."""

    table = []
    for valuation in all_valuations(formula.variables()):
        normalized = {
            name: TruthValue.coerce(value) for name, value in valuation.items()
        }
        table.append((normalized, supervaluation(formula, normalized, semantics)))
    return table


def supervaluation_boolean_pair(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> tuple[dict[BooleanAssignment, bool], dict[BooleanAssignment, bool]]:
    """Return the Boolean pair used in the paper's prime-implicant algorithm.

    The first function is ``sv_true`` and is true exactly on definite
    assignments where the supervaluation is ``TRUE``. The second function is
    ``sv_nonfalse`` and is true exactly on definite assignments where the
    supervaluation is not ``FALSE``.
    """

    variables = sorted(formula.variables())
    sv_true = {}
    sv_nonfalse = {}
    for valuation in classical_valuations(variables):
        key = _normalize_boolean_assignment(
            {name: value is TruthValue.TRUE for name, value in valuation.items()},
            variables,
        )
        value = supervaluation(formula, valuation, semantics)
        sv_true[key] = value is TruthValue.TRUE
        sv_nonfalse[key] = value is not TruthValue.FALSE
    return sv_true, sv_nonfalse


def _oracle_supervaluation_obdd_pair(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> tuple[OBDD, OBDD]:
    """Return OBDDs for the Boolean pair used by the Section 5 algorithm."""

    variables = sorted(formula.variables())

    def sv_true(valuation: Mapping[str, bool]) -> bool:
        return supervaluation(formula, valuation, semantics) is TruthValue.TRUE

    def sv_nonfalse(valuation: Mapping[str, bool]) -> bool:
        return supervaluation(formula, valuation, semantics) is not TruthValue.FALSE

    return (
        build_obdd_from_boolean_function(variables, sv_true),
        build_obdd_from_boolean_function(variables, sv_nonfalse),
    )


def supervaluation_obdd_pair(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> tuple[OBDD, OBDD]:
    """Return OBDDs for the paper's Boolean pair representation.

    Under strong Kleene semantics this uses the direct bottom-up translation of
    Section 5. For other semantics it falls back to the completion-based oracle.
    """

    if semantics == STRONG_KLEENE:
        pair = direct_supervaluation_obdd_pair(formula, semantics)
        return pair.is_true, pair.is_nonfalse
    return _oracle_supervaluation_obdd_pair(formula, semantics)


def enumerate_prime_implicants(
    function_values: BooleanFunctionInput,
    variables: Optional[Sequence[str]] = None,
) -> list[PartialBooleanAssignment]:
    """Enumerate prime implicants of a total Boolean function.

    The input may be either a mapping from complete Boolean assignments to
    output bits or an ``OBDD`` representing the function.
    """

    obdd = _as_obdd(function_values, variables)
    if not isinstance(obdd.ordering, ListOrdering):
        raise TypeError("Prime implicant extraction requires a list-based ordering")
    ordering = cast(ListOrdering, obdd.ordering)
    active_variables = list(ordering.get_list())
    if obdd == 0:
        return []

    reduced_cubes = []
    for path in _enumerate_true_paths(obdd.root):
        normalized = _normalize_cube(path, active_variables)
        reduced_cubes.append(_reduce_implicant_cube(obdd, normalized))

    unique = sorted({tuple(cube) for cube in reduced_cubes}, key=_cube_sort_key)
    prime_implicants = []
    for cube in unique:
        if any(
            other != cube and _cube_subsumes(other, cube) for other in unique
        ):
            continue
        prime_implicants.append(cube)

    return [tuple(cube) for cube in prime_implicants]


def primes_formula(
    function_values: BooleanFunctionInput,
    variables: Optional[Sequence[str]] = None,
) -> Formula:
    """Build the disjunction of all prime implicants of a Boolean function."""

    cubes = enumerate_prime_implicants(function_values, variables)
    return disjoin(*(_partial_to_formula(cube) for cube in cubes))


def constructive_one_formula(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> Formula:
    """Construct Blamey's improved ``One[f]`` formula for supervaluation."""

    variables = sorted(formula.variables())
    terms = []
    assignments = [
        _normalize_threevalued_assignment(valuation, variables)
        for valuation in all_valuations(variables)
    ]
    sv_values = {
        assignment: supervaluation(formula, dict(assignment), semantics)
        for assignment in assignments
    }
    for assignment in assignments:
        if sv_values[assignment] is not TruthValue.TRUE:
            continue
        if any(
            sv_values[refined] is not TruthValue.UNKNOWN
            for refined in assignments
            if _strictly_less_definite(refined, assignment)
        ):
            continue
        terms.append(
            tuple(
                (name, value is TruthValue.TRUE)
                for name, value in assignment
                if value is not TruthValue.UNKNOWN
            )
        )
    return disjoin(*(_partial_to_formula(term) for term in terms))


def constructive_zero_formula(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> Formula:
    """Construct Blamey's improved ``Zero[f]`` formula for supervaluation."""

    variables = sorted(formula.variables())
    clauses = []
    assignments = [
        _normalize_threevalued_assignment(valuation, variables)
        for valuation in all_valuations(variables)
    ]
    sv_values = {
        assignment: supervaluation(formula, dict(assignment), semantics)
        for assignment in assignments
    }
    for assignment in assignments:
        if sv_values[assignment] is not TruthValue.FALSE:
            continue
        if any(
            sv_values[refined] is not TruthValue.UNKNOWN
            for refined in assignments
            if _strictly_less_definite(refined, assignment)
        ):
            continue
        clauses.append(
            tuple(
                (name, value is TruthValue.FALSE)
                for name, value in assignment
                if value is not TruthValue.UNKNOWN
            )
        )
    return conjoin(*(_dual_clause_to_formula(clause) for clause in clauses))


def constructive_semantic_minimize(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> Formula:
    """Minimize via the improved constructive realization from Section 4."""

    return information_join(
        constructive_one_formula(formula, semantics),
        constructive_zero_formula(formula, semantics),
    )


def prime_implicant_semantic_minimize(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> Formula:
    """Minimize via the prime-implicant algorithm from Figure 2."""

    sv_true, sv_nonfalse = supervaluation_obdd_pair(formula, semantics)
    sv_false = ~sv_nonfalse
    return information_join(
        primes_formula(sv_true),
        Not(primes_formula(sv_false)),
    )


def semantically_minimize(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    method: str = "primes",
) -> Formula:
    """Return a semantically minimal variant of ``formula``.

    Supported methods are ``"primes"`` for the Section 5 algorithm and
    ``"constructive"`` for the improved constructive algorithm from Section 4.
    """

    if method == "primes":
        return prime_implicant_semantic_minimize(formula, semantics)
    if method == "constructive":
        return constructive_semantic_minimize(formula, semantics)
    raise ValueError(f"Unsupported minimization method: {method!r}")


def is_semantically_minimal_variant(
    original: Formula,
    candidate: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> bool:
    """Check Definition 3.2 from the paper directly."""

    variables = sorted(original.variables() | candidate.variables())
    for valuation in all_valuations(variables):
        if evaluate(candidate, valuation, semantics) != supervaluation(
            original, valuation, semantics
        ):
            return False
    return True


def improves_precision(
    original: Formula,
    candidate: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> bool:
    """Return whether ``candidate`` is never less definite than ``original``."""

    order = {
        (TruthValue.UNKNOWN, TruthValue.UNKNOWN),
        (TruthValue.UNKNOWN, TruthValue.FALSE),
        (TruthValue.UNKNOWN, TruthValue.TRUE),
        (TruthValue.FALSE, TruthValue.FALSE),
        (TruthValue.TRUE, TruthValue.TRUE),
    }
    variables = sorted(original.variables() | candidate.variables())
    for valuation in all_valuations(variables):
        if (
            evaluate(original, valuation, semantics),
            evaluate(candidate, valuation, semantics),
        ) not in order:
            return False
    return True


__all__ = [
    "BooleanAssignment",
    "PartialBooleanAssignment",
    "BooleanFunctionInput",
    "BooleanPairOBDD",
    "InformationJoinFormula",
    "build_obdd_from_boolean_function",
    "direct_supervaluation_obdd_pair",
    "information_join",
    "supervaluation",
    "supervaluation_truth_table",
    "supervaluation_boolean_pair",
    "supervaluation_obdd_pair",
    "enumerate_prime_implicants",
    "primes_formula",
    "constructive_one_formula",
    "constructive_zero_formula",
    "constructive_semantic_minimize",
    "prime_implicant_semantic_minimize",
    "semantically_minimize",
    "is_semantically_minimal_variant",
    "improves_precision",
]
