from typing import Optional, Any, Tuple


class PySamplerException(Exception):
    """Base class for all the custom exceptions of PySampler"""
    pass


class UnknownSmtLibCommandError(PySamplerException):
    """Raised when the parser finds an unknown command."""
    pass


class SolverReturnedUnknownResultError(PySamplerException):
    """This exception is raised if a solver returns 'unknown' as a result"""
    pass


class UnknownSolverAnswerError(PySamplerException):
    """Raised when a solver returns an invalid response."""
    pass


class NoSolverAvailableError(PySamplerException):
    """No solver is available for the selected Logic."""
    pass


class NonLinearError(PySamplerException):
    """The provided expression is not linear."""
    pass


class UndefinedLogicError(PySamplerException):
    """This exception is raised if an undefined Logic is attempted to be used."""
    pass


class InternalSolverError(PySamplerException):
    """Generic exception to capture errors provided by a solver."""
    pass


class NoLogicAvailableError(PySamplerException):
    """Generic exception to capture errors caused by missing support for logics."""
    pass


class SolverRedefinitionError(PySamplerException):
    """Exception representing errors caused by multiple defintions of solvers
       having the same name."""
    pass


class SolverNotConfiguredForUnsatCoresError(PySamplerException):
    """
    Exception raised if a solver not configured for generating unsat
    cores is required to produce a core.
    """
    pass


class SolverStatusError(PySamplerException):
    """
    Exception raised if a method requiring a specific solver status is
    incorrectly called in the wrong status.
    """
    pass


class ConvertExpressionError(PySamplerException):
    """Exception raised if the converter cannot convert an expression."""

    def __init__(self, message: Optional[str] = None, expression: Optional[Any] = None) -> None:
        super().__init__()
        self.message: Optional[str] = message
        self.expression: Optional[Any] = expression

    def __str__(self) -> str:
        return str(self.message) if self.message else "ConvertExpressionError"


class SolverAPINotFound(PySamplerException):
    """The Python API of the selected solver cannot be found."""
    pass


class UndefinedSymbolError(PySamplerException):
    """The given Symbol is not in the FormulaManager."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name

    def __str__(self) -> str:
        return f"'{self.name}' is not defined!"


class PySamplerModeError(PySamplerException):
    """The current mode is not supported for this operation"""
    pass


class PySamplerImportError(PySamplerException, ImportError):
    pass


class PySamplerValueError(PySamplerException, ValueError):
    pass


class PySamplerTypeError(PySamplerException, TypeError):
    pass


class PySamplerSyntaxError(PySamplerException, SyntaxError):
    def __init__(self, message: str, pos_info: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(message)
        self.pos_info: Optional[Tuple[int, int]] = pos_info
        self.message: str = message

    def __str__(self) -> str:
        if self.pos_info:
            return f"Line {self.pos_info[0]}, Col {self.pos_info[1]}: {self.message}"
        else:
            return self.message


class PySamplerIOError(PySamplerException, IOError):
    pass
