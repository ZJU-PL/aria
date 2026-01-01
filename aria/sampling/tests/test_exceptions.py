"""
Unit tests for aria.sampling.exceptions module.

Tests for custom exception classes.
"""

import pytest
from aria.sampling.exceptions import (
    PySamplerException,
    NoSolverAvailableError,
    NonLinearError,
    UndefinedLogicError,
    SolverReturnedUnknownResultError,
    ConvertExpressionError,
    UndefinedSymbolError,
)


class TestExceptions:
    """Test cases for custom exceptions."""

    def test_base_exception(self):
        """Test base PySamplerException."""
        with pytest.raises(PySamplerException):
            raise PySamplerException("Test error")

    def test_no_solver_available_error(self):
        """Test NoSolverAvailableError."""
        with pytest.raises(NoSolverAvailableError):
            raise NoSolverAvailableError("No solver for logic")

    def test_non_linear_error(self):
        """Test NonLinearError."""
        with pytest.raises(NonLinearError):
            raise NonLinearError("Expression is not linear")

    def test_undefined_logic_error(self):
        """Test UndefinedLogicError."""
        with pytest.raises(UndefinedLogicError):
            raise UndefinedLogicError("Logic not defined")

    def test_solver_returned_unknown_error(self):
        """Test SolverReturnedUnknownResultError."""
        with pytest.raises(SolverReturnedUnknownResultError):
            raise SolverReturnedUnknownResultError("Solver returned unknown")

    def test_convert_expression_error(self):
        """Test ConvertExpressionError with message."""
        err = ConvertExpressionError(message="Cannot convert", expression="x + y")
        assert str(err) == "Cannot convert"
        assert err.expression == "x + y"

    def test_convert_expression_error_without_message(self):
        """Test ConvertExpressionError without message."""
        err = ConvertExpressionError()
        assert str(err) == "ConvertExpressionError"

    def test_undefined_symbol_error(self):
        """Test UndefinedSymbolError."""
        err = UndefinedSymbolError("my_var")
        assert err.name == "my_var"
        assert str(err) == "'my_var' is not defined!"


if __name__ == "__main__":
    pytest.main()
