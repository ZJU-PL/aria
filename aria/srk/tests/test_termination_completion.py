"""
Tests for the completed termination.py module.

This tests the ranking function synthesis features.
"""

import pytest
from fractions import Fraction

from aria.srk.syntax import Context, Type
from aria.srk.termination import (
    TerminationAnalyzer,
    LinearRankingFunction,
    TerminationResult,
    TerminationLLRF,
    make_termination_analyzer
)
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


class TestTerminationLLRF:
    """Test lexicographic ranking function synthesis."""

    def test_create_llrf_synthesizer(self):
        """Test creating LLRF synthesizer."""
        ctx = Context()
        llrf = TerminationLLRF(ctx)

        assert llrf.context == ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
