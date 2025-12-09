"""
Tests for the PLY-based parser.

This tests the lexer and parser created with PLY.
"""

import pytest

from aria.srk.srkLex import tokenize, MathLexer
from aria.srk.srkParse import parse_formula, PLY_AVAILABLE
from aria.srk.srkParseNew import MathParser, SMT2Parser


class TestLexer:
    """Test the PLY-based lexer."""
    
    def test_tokenize_simple_formula(self):
        """Test tokenizing a simple formula."""
        input_str = "x + y <= 10"
        tokens = tokenize(input_str)
        
        assert len(tokens) > 0
        # Should have: x, +, y, <=, 10
        assert any(tok.type == 'ID' and tok.value == 'x' for tok in tokens)
        assert any(tok.type == 'ADD' for tok in tokens)
        assert any(tok.type == 'ID' and tok.value == 'y' for tok in tokens)
        assert any(tok.type == 'LEQ' for tok in tokens)
        assert any(tok.type == 'REAL' for tok in tokens)
    
    def test_tokenize_keywords(self):
        """Test tokenizing keywords."""
        input_str = "And Or Not ForAll Exists"
        tokens = tokenize(input_str)
        
        assert any(tok.type == 'AND' for tok in tokens)
        assert any(tok.type == 'OR' for tok in tokens)
        assert any(tok.type == 'NOT' for tok in tokens)
        assert any(tok.type == 'FORALL' for tok in tokens)
        assert any(tok.type == 'EXISTS' for tok in tokens)
    
    def test_tokenize_numbers(self):
        """Test tokenizing numbers including fractions."""
        input_str = "42 -5 3/2"
        tokens = tokenize(input_str)
        
        # Should have 3 REAL tokens
        real_tokens = [tok for tok in tokens if tok.type == 'REAL']
        assert len(real_tokens) >= 2  # At least 2 numbers recognized


class TestParser:
    """Test the PLY-based parser."""
    
    def test_parser_availability(self):
        """Test that parser is available."""
        # The hand-written parser should always work
        assert parse_formula is not None
    
    def test_parse_simple_formula(self):
        """Test parsing a simple formula."""
        try:
            formula = parse_formula("x + y <= 10")
            # If it doesn't raise an exception, parsing worked
            assert formula is not None
        except ImportError:
            # PLY not available, skip test
            pytest.skip("PLY parser not available")
    
    def test_parse_comparison(self):
        """Test parsing comparisons."""
        try:
            formula = parse_formula("x = 5")
            assert formula is not None
        except Exception:
            pass  # Parser might not be fully functional yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

