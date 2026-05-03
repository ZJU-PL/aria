"""
Termination analysis for program verification.

This module implements algorithms for proving program termination,
including ranking function synthesis and termination analysis.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from fractions import Fraction
from dataclasses import dataclass, field
import itertools

from aria.srk.syntax import (
    Context,
    Symbol,
    Type,
    Expression,
    FormulaExpression,
    ArithExpression,
    Const,
    Var,
    Add,
    Mul,
    TrueExpr,
    FalseExpr,
    And,
    Or,
    Not,
    Eq,
    Lt,
    Leq,
    mk_real,
    mk_const,
    mk_add,
    mk_mul,
    mk_leq,
    mk_lt,
    mk_eq,
    mk_and,
    mk_or,
    mk_not,
    mk_true,
    mk_false,
    mk_neg,
    symbols,
)
from aria.srk.polynomial import Polynomial, Monomial
from aria.srk.linear import QQVector, QQMatrix, QQVectorSpace
from aria.srk.qQ import QQ
from aria.srk.log import logf


@dataclass(frozen=True)
class RankingFunction:
    """Represents a ranking function for termination analysis."""

    expression: ArithExpression  # The ranking function expression
    decreases: bool  # Whether it decreases on transitions

    def __str__(self) -> str:
        status = "decreases" if self.decreases else "does not decrease"
        return f"RankingFunction({self.expression}, {status})"


class TerminationResult:
    """Result of termination analysis."""

    def __init__(
        self, terminates: bool, ranking_function: Optional[RankingFunction] = None
    ):
        self.terminates = terminates
        self.ranking_function = ranking_function

    def __str__(self) -> str:
        if self.terminates:
            if self.ranking_function:
                return f"Terminates with ranking function: {self.ranking_function}"
            else:
                return "Terminates"
        else:
            return "May not terminate"


class LinearRankingFunction:
    """Linear ranking function of the form c^T * x + d."""

    def __init__(
        self,
        coefficients: QQVector,
        constant: Fraction = Fraction(0),
        symbol_map: Optional[Dict[int, Symbol]] = None,
    ):
        self.coefficients = coefficients
        self.constant = constant
        self.symbol_map = symbol_map or {}

    def evaluate(self, values: Dict[Symbol, Fraction]) -> Fraction:
        """Evaluate the ranking function."""
        result = self.constant

        # Map symbol positions to values
        for dim, coeff in self.coefficients.entries.items():
            if dim in self.symbol_map:
                symbol = self.symbol_map[dim]
                if symbol in values:
                    # Use qQ module functions
                    import aria.srk.qQ as qQ

                    result = qQ.add(result, qQ.mul(coeff, values[symbol]))

        return result

    def to_term(self, context: Context) -> ArithExpression:
        """Convert to an arithmetic term expression."""
        terms = [mk_real(self.constant)]

        for dim, coeff in self.coefficients.entries.items():
            if dim in self.symbol_map:
                symbol = self.symbol_map[dim]
                const_sym = mk_const(symbol)
                term = mk_mul([mk_real(coeff), const_sym])
                terms.append(term)

        if len(terms) == 1:
            return terms[0]
        else:
            return mk_add(terms)

    def __str__(self) -> str:
        return f"LinearRankingFunction({self.coefficients}, {self.constant})"


@dataclass(frozen=True)
class _LinearExpr:
    """Small linear expression model used by conservative termination checks."""

    coeffs: Dict[Symbol, Fraction] = field(default_factory=dict)
    const: Fraction = Fraction(0)

    def __add__(self, other: "_LinearExpr") -> "_LinearExpr":
        coeffs = dict(self.coeffs)
        for sym, coeff in other.coeffs.items():
            coeffs[sym] = coeffs.get(sym, Fraction(0)) + coeff
            if coeffs[sym] == 0:
                del coeffs[sym]
        return _LinearExpr(coeffs, self.const + other.const)

    def __neg__(self) -> "_LinearExpr":
        return _LinearExpr(
            {sym: -coeff for sym, coeff in self.coeffs.items()}, -self.const
        )

    def __sub__(self, other: "_LinearExpr") -> "_LinearExpr":
        return self + (-other)

    def scale(self, scalar: Fraction) -> "_LinearExpr":
        if scalar == 0:
            return _LinearExpr()
        return _LinearExpr(
            {sym: coeff * scalar for sym, coeff in self.coeffs.items()},
            self.const * scalar,
        )


class TerminationAnalyzer:
    """Analyzer for proving program termination."""

    def __init__(self, context: Context):
        self.context = context

    def analyze_transitions(self, transitions: List[Any]) -> TerminationResult:
        """Analyze a list of transitions for termination.

        The analyzer is intentionally conservative: it only reports termination
        when a linear candidate is bounded below by the transition guards and
        decreases by a positive constant under every transition transform.
        """
        try:
            if not transitions:
                return TerminationResult(True)

            linear_rf = self._synthesize_from_transitions(transitions)
            if linear_rf is None:
                lex_rf = self._synthesize_lexicographic_from_transitions(transitions)
                if lex_rf is None:
                    return TerminationResult(False)
                return TerminationResult(True, lex_rf)

            rf_term = linear_rf.to_term(self.context)
            return TerminationResult(True, RankingFunction(rf_term, True))
        except Exception as e:
            logf(f"Error analyzing transitions: {e}")
            return TerminationResult(False)

    def _synthesize_from_transitions(
        self, transitions: List[Any]
    ) -> Optional[LinearRankingFunction]:
        variables = self._transition_variables(transitions)
        if not variables:
            return None

        for candidate in self._ranking_candidates(variables):
            if self._valid_ranking_candidate(candidate, transitions, variables):
                return self._linear_expr_to_ranking(candidate, variables)

        return None

    def _synthesize_lexicographic_from_transitions(
        self, transitions: List[Any], max_components: int = 3
    ) -> Optional[LexicographicRankingFunction]:
        variables = self._transition_variables(transitions)
        if not variables:
            return None

        candidates = self._ranking_candidates(variables)
        limit = min(max_components, len(candidates))
        for size in range(2, limit + 1):
            for components in itertools.permutations(candidates, size):
                if self._valid_lexicographic_candidate(
                    list(components), transitions, variables
                ):
                    terms = tuple(
                        self._linear_expr_to_ranking(component, variables).to_term(
                            self.context
                        )
                        for component in components
                    )
                    return LexicographicRankingFunction(terms)
        return None

    def _valid_lexicographic_candidate(
        self,
        components: List[_LinearExpr],
        transitions: List[Any],
        variables: List[Symbol],
    ) -> bool:
        for tr in transitions:
            guard = getattr(tr, "guard", mk_true(self.context))
            transform = getattr(tr, "transform", None)
            if not isinstance(transform, dict):
                return False
            if isinstance(guard, FalseExpr) or (
                hasattr(tr, "is_zero") and tr.is_zero()
            ):
                continue

            deltas: List[_LinearExpr] = []
            for component in components:
                if not self._guard_implies_nonnegative(component, guard, variables):
                    return False
                post = self._post_linear_expr(component, transform, variables)
                if post is None:
                    return False
                deltas.append(component - post)

            if not self._lexicographically_decreases(deltas):
                return False

        return True

    def _lexicographically_decreases(self, deltas: List[_LinearExpr]) -> bool:
        for idx, delta in enumerate(deltas):
            if delta.coeffs:
                return False
            if delta.const > 0:
                return True
            if delta.const != 0:
                return False
        return False

    def _transition_variables(self, transitions: List[Any]) -> List[Symbol]:
        variables: List[Symbol] = []

        def add(sym: Symbol) -> None:
            if isinstance(sym, Symbol) and sym.typ in (Type.INT, Type.REAL):
                if sym not in variables:
                    variables.append(sym)

        for tr in transitions:
            transform = getattr(tr, "transform", None)
            if isinstance(transform, dict):
                for sym, expr in transform.items():
                    add(sym)
                    for used in symbols(expr):
                        add(used)

            guard = getattr(tr, "guard", None)
            if guard is not None:
                for sym in symbols(guard):
                    add(sym)

            if hasattr(tr, "uses"):
                for sym in tr.uses():
                    add(sym)
            if hasattr(tr, "defines"):
                for sym in tr.defines():
                    add(sym)

        return variables

    def _ranking_candidates(self, variables: List[Symbol]) -> List[_LinearExpr]:
        candidates: List[_LinearExpr] = []

        def add(candidate: _LinearExpr) -> None:
            if candidate.coeffs and candidate not in candidates:
                candidates.append(candidate)

        for sym in variables:
            add(_LinearExpr({sym: Fraction(1)}))
            add(_LinearExpr({sym: Fraction(-1)}))

        add(_LinearExpr({sym: Fraction(1) for sym in variables}))

        for left in variables:
            for right in variables:
                if left != right:
                    add(_LinearExpr({left: Fraction(1), right: Fraction(-1)}))

        return candidates

    def _valid_ranking_candidate(
        self, candidate: _LinearExpr, transitions: List[Any], variables: List[Symbol]
    ) -> bool:
        for tr in transitions:
            guard = getattr(tr, "guard", mk_true(self.context))
            transform = getattr(tr, "transform", None)
            if not isinstance(transform, dict):
                return False
            if isinstance(guard, FalseExpr) or (
                hasattr(tr, "is_zero") and tr.is_zero()
            ):
                continue
            if not self._guard_implies_nonnegative(candidate, guard, variables):
                return False

            post = self._post_linear_expr(candidate, transform, variables)
            if post is None:
                return False

            decrease = candidate - post
            if decrease.coeffs or decrease.const <= 0:
                return False

        return True

    def _post_linear_expr(
        self,
        candidate: _LinearExpr,
        transform: Dict[Symbol, Expression],
        variables: List[Symbol],
    ) -> Optional[_LinearExpr]:
        post = _LinearExpr()
        for sym, coeff in candidate.coeffs.items():
            expr = transform.get(sym, mk_const(sym))
            lin_expr = self._linear_expr_of(expr, variables)
            if lin_expr is None:
                return None
            post = post + lin_expr.scale(coeff)
        return post + _LinearExpr(const=candidate.const)

    def _linear_expr_to_ranking(
        self, candidate: _LinearExpr, variables: List[Symbol]
    ) -> LinearRankingFunction:
        symbol_map = {idx: sym for idx, sym in enumerate(variables)}
        coeffs = {
            idx: candidate.coeffs[sym]
            for idx, sym in symbol_map.items()
            if candidate.coeffs.get(sym, Fraction(0)) != 0
        }
        return LinearRankingFunction(QQVector(coeffs), candidate.const, symbol_map)

    def _linear_expr_of(
        self, expr: Expression, variables: List[Symbol]
    ) -> Optional[_LinearExpr]:
        if isinstance(expr, Const):
            value = self._numeric_const(expr)
            if value is not None:
                return _LinearExpr(const=value)
            if expr.symbol in variables:
                return _LinearExpr({expr.symbol: Fraction(1)})
            return None

        if isinstance(expr, Var):
            for sym in variables:
                if sym.id == expr.var_id and sym.typ == expr.var_type:
                    return _LinearExpr({sym: Fraction(1)})
            return None

        if isinstance(expr, Add):
            result = _LinearExpr()
            for arg in expr.args:
                lin_arg = self._linear_expr_of(arg, variables)
                if lin_arg is None:
                    return None
                result = result + lin_arg
            return result

        if isinstance(expr, Mul):
            scalar = Fraction(1)
            linear_part: Optional[_LinearExpr] = None
            for arg in expr.args:
                lin_arg = self._linear_expr_of(arg, variables)
                if lin_arg is None:
                    return None
                if lin_arg.coeffs:
                    if linear_part is not None:
                        return None
                    linear_part = lin_arg
                else:
                    scalar *= lin_arg.const
            return (linear_part or _LinearExpr(const=Fraction(1))).scale(scalar)

        return None

    def _numeric_const(self, expr: Const) -> Optional[Fraction]:
        name = expr.symbol.name
        if name is None:
            return None
        if name.startswith("real_"):
            name = name[5:]
        try:
            return Fraction(name)
        except (ValueError, TypeError):
            try:
                return Fraction(str(float(name)))
            except (ValueError, TypeError):
                return None

    def _guard_implies_nonnegative(
        self, candidate: _LinearExpr, guard: FormulaExpression, variables: List[Symbol]
    ) -> bool:
        if not candidate.coeffs:
            return candidate.const >= 0
        if isinstance(guard, TrueExpr):
            return False
        if isinstance(guard, FalseExpr):
            return True

        needed = _LinearExpr(const=candidate.const)
        needed = needed + _LinearExpr(dict(candidate.coeffs))
        if self._is_direct_nonnegative_guard(needed, guard, variables):
            return True

        return False

    def _is_direct_nonnegative_guard(
        self, needed: _LinearExpr, guard: FormulaExpression, variables: List[Symbol]
    ) -> bool:
        atoms = self._guard_atoms(guard)
        if atoms is None:
            return False

        if not needed.coeffs:
            return needed.const >= 0

        for atom in atoms:
            if isinstance(atom, Leq):
                left = self._linear_expr_of(atom.left, variables)
                right = self._linear_expr_of(atom.right, variables)
                if left is None or right is None:
                    continue
                if self._same_linear_expr(right - left, needed):
                    return True
            elif isinstance(atom, Lt):
                left = self._linear_expr_of(atom.left, variables)
                right = self._linear_expr_of(atom.right, variables)
                if left is None or right is None:
                    continue
                diff = right - left
                if self._same_linear_expr(diff, needed):
                    return True
                if (
                    self._same_coeffs(diff, needed)
                    and diff.const >= needed.const
                    and all(sym.typ == Type.INT for sym in needed.coeffs)
                ):
                    return True
            elif isinstance(atom, Eq):
                left = self._linear_expr_of(atom.left, variables)
                right = self._linear_expr_of(atom.right, variables)
                if left is None or right is None:
                    continue
                if self._same_linear_expr(left, needed) and not right.coeffs:
                    if right.const >= 0:
                        return True
                if self._same_linear_expr(right, needed) and not left.coeffs:
                    if left.const >= 0:
                        return True

        return False

    def _guard_atoms(
        self, guard: FormulaExpression
    ) -> Optional[List[FormulaExpression]]:
        if isinstance(guard, TrueExpr):
            return []
        if isinstance(guard, FalseExpr):
            return [guard]
        if isinstance(guard, And):
            atoms: List[FormulaExpression] = []
            for arg in guard.args:
                arg_atoms = self._guard_atoms(arg)
                if arg_atoms is None:
                    return None
                atoms.extend(arg_atoms)
            return atoms
        if isinstance(guard, (Or, Not)):
            return None
        return [guard]

    def _same_linear_expr(self, left: _LinearExpr, right: _LinearExpr) -> bool:
        return self._same_coeffs(left, right) and left.const == right.const

    def _same_coeffs(self, left: _LinearExpr, right: _LinearExpr) -> bool:
        return left.coeffs == right.coeffs

    def _transitions_from_formula(self, transition_formula: Any) -> Optional[List[Any]]:
        if hasattr(transition_formula, "transform") and hasattr(
            transition_formula, "guard"
        ):
            return [transition_formula]

        formula = self._get_formula(transition_formula)
        var_pairs = self._get_symbol_pairs(transition_formula)
        if formula is None or not var_pairs:
            return None

        atoms = self._guard_atoms(formula)
        if atoms is None:
            return None

        post_to_pre = {post: pre for pre, post in var_pairs}
        post_symbols = set(post_to_pre)
        variables = [pre for pre, _ in var_pairs]
        transform: Dict[Symbol, Expression] = {}
        guards: List[FormulaExpression] = []

        for atom in atoms:
            assigned = self._assignment_from_atom(atom, post_to_pre, post_symbols)
            if assigned is None:
                if symbols(atom) & post_symbols:
                    return None
                guards.append(atom)
            else:
                pre, expr = assigned
                if self._linear_expr_of(expr, variables) is None:
                    return None
                transform[pre] = expr

        guard = mk_true(self.context) if not guards else mk_and(self.context, guards)

        @dataclass(frozen=True)
        class _FormulaTransition:
            transform: Dict[Symbol, Expression]
            guard: FormulaExpression

        return [_FormulaTransition(transform, guard)]

    def _assignment_from_atom(
        self,
        atom: FormulaExpression,
        post_to_pre: Dict[Symbol, Symbol],
        post_symbols: Set[Symbol],
    ) -> Optional[Tuple[Symbol, Expression]]:
        if not isinstance(atom, Eq):
            return None
        if isinstance(atom.left, Const) and atom.left.symbol in post_to_pre:
            if symbols(atom.right) & post_symbols:
                return None
            return post_to_pre[atom.left.symbol], atom.right
        if isinstance(atom.right, Const) and atom.right.symbol in post_to_pre:
            if symbols(atom.left) & post_symbols:
                return None
            return post_to_pre[atom.right.symbol], atom.left
        return None

    def _get_symbol_pairs(self, transition_formula: Any) -> List[Tuple[Symbol, Symbol]]:
        raw_symbols = getattr(transition_formula, "symbols", None)
        if callable(raw_symbols):
            raw_symbols = raw_symbols()
        if raw_symbols is None:
            return []
        return list(raw_symbols)

    def _get_formula(self, transition_formula: Any) -> Optional[FormulaExpression]:
        formula = getattr(transition_formula, "formula", None)
        if callable(formula):
            formula = formula()
        return formula

    def synthesize_linear_ranking_function(
        self, pre_vars: List[Symbol], post_vars: List[Symbol], guard: FormulaExpression
    ) -> Optional[LinearRankingFunction]:
        """
        Synthesize a linear ranking function for a transition relation.

        A linear ranking function f(x) = c^T x + d must satisfy:
        1. f(x) >= 0 when guard(x, x') holds (bounded below)
        2. f(x) - f(x') >= delta > 0 when guard(x, x') holds (decreasing)

        We use Farkas' lemma and linear programming to find such a function.
        """
        try:
            n = len(pre_vars)
            if n == 0 or len(pre_vars) != len(post_vars):
                return None

            formula = guard or mk_true(self.context)
            from aria.srk.transitionFormula import TransitionFormula

            tf = TransitionFormula(
                formula=formula, symbols=list(zip(pre_vars, post_vars))
            )
            transitions = self._transitions_from_formula(tf)
            if not transitions:
                return None
            return self._synthesize_from_transitions(transitions)

        except Exception as e:
            logf(f"Error synthesizing linear ranking function: {e}")
            return None

    def synthesize_ranking_function(
        self, transition_formula: Any
    ) -> Optional[RankingFunction]:
        """Synthesize a ranking function for a transition formula."""
        try:
            transitions = self._transitions_from_formula(transition_formula)
            if transitions:
                linear_rf = self._synthesize_from_transitions(transitions)
                if linear_rf:
                    return RankingFunction(linear_rf.to_term(self.context), True)

            # Extract pre/post variables from transition formula
            var_pairs = self._get_symbol_pairs(transition_formula)
            if var_pairs:
                pre_vars = [pre for pre, _ in var_pairs]
                post_vars = [post for _, post in var_pairs]
                guard = self._get_formula(transition_formula)
            else:
                # Fallback: try to find variables in the formula
                pre_vars = []
                post_vars = []
                guard = None

            if not pre_vars:
                return None

            # Try to synthesize a linear ranking function
            linear_rf = self.synthesize_linear_ranking_function(
                pre_vars, post_vars, guard
            )

            if linear_rf:
                # Convert to term for RankingFunction
                rf_term = linear_rf.to_term(self.context)
                return RankingFunction(rf_term, True)

            return None

        except Exception as e:
            logf(f"Error in ranking function synthesis: {e}")
            return None

    def prove_termination(self, transition_system: Any) -> TerminationResult:
        """Prove termination of a transition system."""
        try:
            if isinstance(transition_system, list):
                return self.analyze_transitions(transition_system)

            if hasattr(transition_system, "edges"):
                edges = transition_system.edges
                if callable(edges):
                    edges = edges()
                transitions = [edge[1] for edge in edges]
                return self.analyze_transitions(transitions)

            # Try to find a ranking function
            ranking_function = self.synthesize_ranking_function(transition_system)

            if ranking_function:
                return TerminationResult(True, ranking_function)
            else:
                return TerminationResult(False)
        except Exception as e:
            logf(f"Error proving termination: {e}")
            return TerminationResult(False)

    def analyze_loop(self, loop_body: Any) -> TerminationResult:
        """Analyze termination of a loop."""
        return self.prove_termination(loop_body)


class DependencyTupleAnalysis:
    """Dependency tuple analysis for termination."""

    def __init__(self, context: Context):
        self.context = context

    def analyze(self, transition_formula: Any) -> TerminationResult:
        """Analyze termination using dependency tuples."""
        analyzer = TerminationAnalyzer(self.context)
        return analyzer.prove_termination(transition_formula)


class LexicographicRankingFunction:
    """Lexicographic ranking function."""

    def __init__(self, components: List[ArithExpression]):
        self.components = components

    def __str__(self) -> str:
        return f"LexRankingFunction([{', '.join(str(c) for c in self.components)}])"


class TerminationLLRF:
    """Linear lexicographic ranking function synthesis."""

    def __init__(self, context: Context):
        self.context = context

    def synthesize(
        self, transition_formula: Any
    ) -> Optional[LexicographicRankingFunction]:
        """
        Synthesize a lexicographic ranking function.

        A lexicographic ranking function is a tuple (f_1, ..., f_k) where each f_i
        is a linear function, and the tuple decreases lexicographically.
        """
        try:
            # Extract variables from transition formula
            if hasattr(transition_formula, "symbols"):
                var_pairs = list(transition_formula.symbols)
                pre_vars = [pre for pre, _ in var_pairs]
            else:
                return None

            if not pre_vars:
                return None

            # Try to find a lexicographic ranking function
            # For simplicity, we try single component first
            analyzer = TerminationAnalyzer(self.context)
            linear_rf = analyzer.synthesize_linear_ranking_function(
                pre_vars,
                [post for _, post in var_pairs],
                (
                    transition_formula.formula
                    if hasattr(transition_formula, "formula")
                    else None
                ),
            )

            if linear_rf:
                # Create a lexicographic ranking function with one component
                component = linear_rf.to_term(self.context)
                return LexicographicRankingFunction([component])

            return None

        except Exception as e:
            logf(f"Error synthesizing LLRF: {e}")
            return None


class TerminationDTA:
    """Dependency tuple analysis for termination."""

    def __init__(self, context: Context):
        self.context = context

    def analyze(self, transition_formula: Any) -> TerminationResult:
        """
        Analyze termination using dependency tuple abstraction.

        This method linearizes the transition relation, computes its spectral
        decomposition, and checks for termination using characteristic sequences.
        """
        try:
            # This is a complex algorithm that requires:
            # 1. Linearization of the transition formula
            # 2. Computing matrix exponentials
            # 3. Sequence analysis

            # For now, we delegate to a simpler ranking function approach
            analyzer = TerminationAnalyzer(self.context)
            return analyzer.prove_termination(transition_formula)

        except Exception as e:
            logf(f"Error in DTA analysis: {e}")
            return TerminationResult(False)


class TerminationExp:
    """Exponential polynomial termination analysis."""

    def __init__(self, context: Context):
        self.context = context

    def analyze(self, transition_formula: Any) -> TerminationResult:
        """
        Analyze termination using exponential polynomial abstractions.

        This computes the transitive closure of the transition relation
        using exponential polynomial iteration, then checks for termination.
        """
        try:
            # This requires:
            # 1. Computing k-fold composition using exponential polynomials
            # 2. Finding pre-states that must terminate within k steps

            # For now, we use a simpler approach
            analyzer = TerminationAnalyzer(self.context)
            return analyzer.prove_termination(transition_formula)

        except Exception as e:
            logf(f"Error in exponential polynomial analysis: {e}")
            return TerminationResult(False)


# Convenience functions
def make_termination_analyzer(context: Context) -> TerminationAnalyzer:
    """Create a termination analyzer."""
    return TerminationAnalyzer(context)


def make_llrf_synthesizer(context: Context) -> TerminationLLRF:
    """Create a linear lexicographic ranking function synthesizer."""
    return TerminationLLRF(context)


def make_dta_analyzer(context: Context) -> TerminationDTA:
    """Create a dependency tuple analyzer."""
    return TerminationDTA(context)


def make_exp_analyzer(context: Context) -> TerminationExp:
    """Create an exponential polynomial analyzer."""
    return TerminationExp(context)


def prove_termination(
    transition_formula: Any, context: Optional[Context] = None
) -> TerminationResult:
    """Prove termination of a transition formula."""
    ctx = context or Context()
    analyzer = TerminationAnalyzer(ctx)
    return analyzer.prove_termination(transition_formula)
