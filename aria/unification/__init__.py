"""Unification library for Python.

This module provides logic variable unification and pattern matching capabilities.
"""

from .core import assoc, reify, unify
from .more import unifiable
from .variable import Var, isvar, var, variables, vars as vars_  # noqa: F401
