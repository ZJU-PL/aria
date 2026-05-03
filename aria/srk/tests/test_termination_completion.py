"""
Tests for the completed termination.py module.

This tests the ranking function synthesis features.
"""

import pytest
from fractions import Fraction

from aria.srk.syntax import (
    Context,
    Type,
    mk_add,
    mk_and,
    mk_const,
    mk_eq,
    mk_lt,
    mk_real,
)
from aria.srk.termination import (
    TerminationAnalyzer,
    LinearRankingFunction,
    TerminationResult,
    TerminationLLRF,
    make_termination_analyzer,
)
from aria.srk.terminationLLRF import analyze_termination_llrf, has_llrf
from aria.srk.transitionFormula import TransitionFormula
from aria.srk.transition import Transition
from aria.srk.linear import QQVector
from aria.srk import qQ as QQ


class TestLinearRankingFunction:
    """Test linear ranking function representation."""

    def test_create_linear_ranking_function(self):
        """Test creating a linear ranking function."""
        # Create f(x) = -x + 5
        coeffs = QQVector({0: Fraction(-1)})
        const = Fraction(5)

        rf = LinearRankingFunction(coeffs, const)

        assert rf.coefficients.entries[0] == Fraction(-1)
        assert rf.constant == Fraction(5)

    def test_linear_ranking_function_evaluation(self):
        """Test evaluating a linear ranking function."""
        from aria.srk.syntax import mk_symbol

        ctx = Context()
        x = mk_symbol("x", Type.REAL)

        # Create f(x) = -x + 5
        coeffs = QQVector({0: Fraction(-1)})
        const = Fraction(5)
        symbol_map = {0: x}

        rf = LinearRankingFunction(coeffs, const, symbol_map)

        # Evaluate at x=2: -2 + 5 = 3
        result = rf.evaluate({x: Fraction(2)})
        assert result == Fraction(3)

    def test_linear_ranking_function_to_term(self):
        """Test converting ranking function to term."""
        from aria.srk.syntax import mk_symbol

        ctx = Context()
        x = mk_symbol("x", Type.REAL)

        # Create f(x) = -x + 5
        coeffs = QQVector({0: Fraction(-1)})
        const = Fraction(5)
        symbol_map = {0: x}

        rf = LinearRankingFunction(coeffs, const, symbol_map)
        term = rf.to_term(ctx)

        # Just verify it creates a term
        assert term is not None


class TestTerminationAnalyzer:
    """Test termination analyzer."""

    def test_create_analyzer(self):
        """Test creating a termination analyzer."""
        ctx = Context()
        analyzer = TerminationAnalyzer(ctx)

        assert analyzer.context == ctx

    def test_analyzer_factory(self):
        """Test factory function for analyzer."""
        ctx = Context()
        analyzer = make_termination_analyzer(ctx)

        assert isinstance(analyzer, TerminationAnalyzer)

    def test_termination_result(self):
        """Test termination result structure."""
        result = TerminationResult(True)
        assert result.terminates == True
        assert str(result) == "Terminates"

        result2 = TerminationResult(False)
        assert result2.terminates == False
        assert str(result2) == "May not terminate"

    def test_transition_formula_decrementing_counter(self):
        """TransitionFormula equalities are converted to guarded transforms."""
        ctx = Context()
        analyzer = TerminationAnalyzer(ctx)
        x = ctx.mk_symbol("x", Type.INT)
        xp = ctx.mk_symbol("x'", Type.INT)
        zero = mk_real(ctx, QQ.zero())
        minus_one = mk_real(ctx, QQ.of_int(-1))
        formula = mk_and(
            ctx,
            [
                mk_lt(zero, mk_const(x)),
                mk_eq(mk_const(xp), mk_add([mk_const(x), minus_one])),
            ],
        )
        tf = TransitionFormula(formula=formula, symbols=[(x, xp)])

        result = analyzer.prove_termination(tf)

        assert result.terminates is True
        assert result.ranking_function is not None

    def test_transition_formula_with_unsupported_post_guard_is_unknown(self):
        """Post-state guards are not silently treated as pre-state bounds."""
        ctx = Context()
        analyzer = TerminationAnalyzer(ctx)
        x = ctx.mk_symbol("x", Type.INT)
        xp = ctx.mk_symbol("x'", Type.INT)
        zero = mk_real(ctx, QQ.zero())
        minus_one = mk_real(ctx, QQ.of_int(-1))
        formula = mk_and(
            ctx,
            [
                mk_lt(zero, mk_const(xp)),
                mk_eq(mk_const(xp), mk_add([mk_const(x), minus_one])),
            ],
        )
        tf = TransitionFormula(formula=formula, symbols=[(x, xp)])

        result = analyzer.prove_termination(tf)

        assert result.terminates is False


class TestTerminationLLRF:
    """Test lexicographic ranking function synthesis."""

    def test_create_llrf_synthesizer(self):
        """Test creating LLRF synthesizer."""
        ctx = Context()
        llrf = TerminationLLRF(ctx)

        assert llrf.context == ctx

    def test_standalone_llrf_placeholders_are_non_proofs(self):
        """The incomplete standalone LLRF API must not return unsound success."""
        ctx = Context()
        x = ctx.mk_symbol("x", Type.INT)
        xp = ctx.mk_symbol("x'", Type.INT)
        formula = mk_eq(mk_const(xp), mk_const(x))
        tf = TransitionFormula(formula=formula, symbols=[(x, xp)])
        transition = Transition.assign(ctx, x, mk_const(x))

        assert has_llrf(ctx, tf) is False
        assert analyze_termination_llrf([transition], ctx) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
