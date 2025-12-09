"""
SYMBA: Symbolic Optimization with SMT Solvers

This module implements the SYMBA algorithm from the paper "Symbolic Optimization with SMT Solvers， POPL 2014"
for optimizing objective functions in linear real arithmetic using SMT solvers as black boxes.
"""

from .symba import SYMBA, SYMBAState, InferenceRule
from .multi_symba import MultiSYMBA

__all__ = ['SYMBA', 'SYMBAState', 'InferenceRule', 'MultiSYMBA']
