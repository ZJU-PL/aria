# Python 3.13 compatibility fix for six.moves
# Some six releases cannot build ``six.moves`` lazily on Python 3.13. Only
# install a fallback when the real ``six.moves`` is unusable: replacing a
# working ``six.moves`` hides every attribute six registers (``range``,
# ``StringIO``, ...) and breaks downstream users such as
# ``pysat.examples.rc2``/``hitman``.
import sys

if sys.version_info >= (3, 13):
    try:
        import six
        import types
        import configparser
        import io

        try:
            from six.moves import range as _six_moves_range  # noqa: F401
            from six.moves import configparser as _six_moves_cfg  # noqa: F401
        except ImportError:
            # Create a real module for six.moves and register it in sys.modules
            # This allows "import six.moves.*" to work
            moves_module = types.ModuleType("six.moves")
            sys.modules["six.moves"] = moves_module
            six.moves = moves_module

            # Add commonly needed modules from six.moves
            moves_module.configparser = configparser
            sys.modules["six.moves.configparser"] = configparser

            # xrange was removed in Python 3, it's just range now
            moves_module.xrange = range
            moves_module.range = range

            # cStringIO was removed in Python 3, use io.StringIO instead
            moves_module.cStringIO = io.StringIO
    except ImportError:
        pass  # six not installed yet, will fail later with a clearer error

import os

# Version detection
from importlib.metadata import version as _version
try:
    __version__ = _version("pyaria")
except Exception:
    try:
        __version__ = _version("aria")
    except Exception:
        __version__ = "0.2.0"

# Debug flag - can be set via environment variable ARIA_DEBUG
ARIA_DEBUG = os.environ.get("ARIA_DEBUG", "False").lower() in ("true", "1", "yes")

# Public API exports
__all__ = [
    # Core modules
    "utils",         # Utilities (including srk, solver, translator, etc.)
    "smt",           # SMT operations and utilities
    "bool",          # Boolean operations and engines
    "quant",         # Quantifier reasoning and solvers
    "pyomt",         # Optimization and MaxSAT solvers
    # Specialized modules
    "proof",         # Proof utilities (abduction, interpolant, unsat_core)
    "allsmt",        # AllSMT (enumerate all satisfying models)
    "automata",      # Automata operations
    "backbone",      # Backbone literal computation
    "concolic",      # Native Python concolic testing
    "counting",      # Model counting
    "datalog",       # Datalog and logic-programming components
    "efmc",          # Program verification frontends and engines
    "ml",            # Machine learning and LLM-assisted reasoning
    "scsat",         # Shared-context batched satisfiability
    "sampling",      # Sampling operations
    "symabs",        # Symbolic abstraction
    "synthesis",     # Program synthesis
    "volumn",        # Volume computation for polytopes
    # CLI
    "cli",           # Command-line interface
]

# Keep subpackages importable without forcing optional dependency trees.
