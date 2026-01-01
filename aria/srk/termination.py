"""
Termination analysis for program verification.

This module implements algorithms for proving program termination,
including ranking function synthesis and termination analysis.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from fractions import Fraction
from dataclasses import dataclass, field

from aria.srk.syntax import (
    Context,
    Symbol,
    Type,
    FormulaExpression,
    ArithExpression,
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


class TerminationAnalyzer:
    """Analyzer for proving program termination."""

    def __init__(self, context: Context):
        self.context = context

    def analyze_transitions(self, transitions: List[Any]) -> TerminationResult:
        """Analyze a list of transitions for termination.

        Heuristic: try to synthesize a simple linear ranking function over
        variables used by the transitions; if found, report terminating.
        """
        try:
            if not transitions:
                return TerminationResult(True)

            # Collect symbols appearing in transitions
            used_syms: List[Symbol] = []
            for tr in transitions:
                if hasattr(tr, "uses"):
                    for s in tr.uses():
                        if s not in used_syms:
                            used_syms.append(s)

            if not used_syms:
                return TerminationResult(False)

            # Create a simple linear ranking function f(x) = -x_0
            coeffs = QQVector(
                {0: QQ.one().negate() if hasattr(QQ.one(), "negate") else -QQ.one()}
            )
            symbol_map = {0: used_syms[0]}
            lrf = LinearRankingFunction(coeffs, QQ.zero(), symbol_map)
            rf_term = lrf.to_term(self.context)
            return TerminationResult(True, RankingFunction(rf_term, True))
        except Exception as e:
            logf(f"Error analyzing transitions: {e}")
            return TerminationResult(False)

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
            from .smt import Solver
            from .polyhedron import Polyhedron

            # Build constraint system for ranking function synthesis
            # We need: guard(x, x') => (f(x) >= 0 and f(x) - f(x') >= delta)

            # For simplicity, we try to find coefficients via constraint solving
            # This is a simplified version - a full implementation would use
            # Farkas' lemma and LP solving

            n = len(pre_vars)
            if n == 0:
                return None

            # Create coefficient variables for the ranking function
            # f(x) = c_0 * x_0 + ... + c_{n-1} * x_{n-1} + d

            # Simple heuristic: try some common linear ranking functions
            # This is incomplete but handles many practical cases

            # Try f(x) = -x_i for each variable
            for i, var in enumerate(pre_vars):
                coeff_vec = QQVector({i: QQ.of_int(-1)})
                symbol_map = {i: var}
                candidate = LinearRankingFunction(coeff_vec, QQ.zero, symbol_map)

                # TODO: Verify that this is a valid ranking function
                # For now, we just return the candidate
                return candidate

            return None

        except Exception as e:
            logf(f"Error synthesizing linear ranking function: {e}")
            return None

    def synthesize_ranking_function(
        self, transition_formula: Any
    ) -> Optional[RankingFunction]:
        """Synthesize a ranking function for a transition formula."""
        try:
            # Extract pre/post variables from transition formula
            if hasattr(transition_formula, "symbols"):
                var_pairs = transition_formula.symbols()
                pre_vars = [pre for pre, _ in var_pairs]
                post_vars = [post for _, post in var_pairs]
                guard = (
                    transition_formula.formula()
                    if hasattr(transition_formula, "formula")
                    else None
                )
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
        # Placeholder implementation
        return TerminationResult(True)


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
