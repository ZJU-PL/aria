"""Exceptions for the CDCL(T) SMT solver"""

class CDCLTError(Exception):
    """Base exception for CDCL(T) solver errors"""
    pass


class TheorySolverError(CDCLTError):
    """Theory solver encountered an error"""
    pass


class PreprocessingError(CDCLTError):
    """Preprocessing phase encountered an error"""
    pass
