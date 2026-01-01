# coding: utf-8
"""
Public subclasses of different Exceptions
"""


class AriaException(Exception):
    """Base class for ARIA exceptions"""

    pass


class SMTSuccess(AriaException):
    """Flag for good state"""

    pass


class SMTError(AriaException):
    """TBD"""

    pass


class SMTUnknown(AriaException):
    """TBD"""

    pass


class SMTLIBSolverError(SMTError):
    """TBD"""

    pass


class UndefinedLogicError(AriaException):
    """This exception is raised if an undefined Logic is attempted to be used."""

    pass


class NoLogicAvailableError(AriaException):
    """Generic exception to capture errors caused by missing support for logics."""

    pass
