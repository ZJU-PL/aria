"""Exceptions for the CDCL(T) SMT solver"""


class CDCLTError(Exception):
    """Base exception for CDCL(T) solver errors"""


class TheorySolverError(CDCLTError):
    """Theory solver encountered an error"""


class PreprocessingError(CDCLTError):
    """Preprocessing phase encountered an error"""
