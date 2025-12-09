# coding: utf-8
"""
Common types for different components of aria
"""
from enum import Enum, auto


class SolverResult(Enum):
    UNSAT = auto()
    UNKNOWN = auto()
    SAT = auto()
    ERROR = auto()


class OSType(Enum):
    LINUX = auto()
    WINDOWS = auto()
    MAC = auto()
    UNKNOWN = auto()


class BinarySMTSolverType(Enum):
    # the binary solver may be used "globally" by different components
    # So, we put it here
    Z3 = auto()
    CVC4 = auto()
    YICES = auto()
    BOOLECTOR = auto()
    UNKNOWN = auto()
