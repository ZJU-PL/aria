"""Datalog-related functionality for ARIA."""

from . import version
from . import aggregate
from . import logic as _logic_module
from . import py_engine
from . import py_parser
from . import user_list

Logic = _logic_module.Logic
logic = _logic_module
Logic().clear()

from . import py_datalog

Aggregate = aggregate
UserList = user_list
pyDatalog = py_datalog
pyEngine = py_engine
pyParser = py_parser
from .api import (
    DatalogParseError,
    DatalogAPIError,
    Function,
    Program,
    QueryResult,
    Relation,
    Rule,
    UndefinedPredicateError,
    Variable,
    vars_,
)
from . import util

__all__ = [
    "Aggregate",
    "DatalogAPIError",
    "DatalogParseError",
    "Function",
    "Logic",
    "Program",
    "QueryResult",
    "Relation",
    "Rule",
    "UndefinedPredicateError",
    "UserList",
    "Variable",
    "aggregate",
    "logic",
    "py_datalog",
    "py_engine",
    "py_parser",
    "pyDatalog",
    "pyEngine",
    "pyParser",
    "user_list",
    "util",
    "vars_",
    "version",
]
