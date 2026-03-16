"""
Custom exception hierarchy for SMT-related errors.
"""


class SMTError(Exception):
    """Top level Exception object for custom exception hierarchy."""


class SmtlibError(SMTError):
    """Exception for SMTLIB-related errors."""


class SolverError(SmtlibError):
    """Exception for solver-related errors."""
