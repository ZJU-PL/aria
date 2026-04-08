"""Datalog-related functionality for ARIA."""

from . import version
from . import Logic as _logic_module

Logic = _logic_module.Logic
Logic().clear()

from . import Aggregate
from . import UserList
from .api import (
    DatalogAPIError,
    Program,
    QueryResult,
    Relation,
    Rule,
    UndefinedPredicateError,
    Variable,
    vars_,
)
from . import pyDatalog
from . import pyEngine
from . import pyParser
from . import util

__all__ = [
    "Aggregate",
    "DatalogAPIError",
    "Logic",
    "Program",
    "QueryResult",
    "Relation",
    "Rule",
    "UndefinedPredicateError",
    "UserList",
    "Variable",
    "pyDatalog",
    "pyEngine",
    "pyParser",
    "util",
    "vars_",
    "version",
]
