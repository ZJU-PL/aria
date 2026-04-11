"""
Quantifier Elimination (QE) module for aria.

This module provides various approaches to quantifier elimination:
- External tools: QEPCAD, Mathematica, Redlog
- Internal algorithms: Shannon expansion, LME-based methods
- Unified interface for external tools
"""

from importlib import import_module

# Import the unified external QE solver
from .external_qe import (
    ExternalQESolver,
    QESolverConfig,
    QEBackend,
    eliminate_quantifiers_qepcad,
    eliminate_quantifiers_mathematica,
    eliminate_quantifiers_redlog,
)

from . import qe_expansion
from . import qe_fm
from . import qe_lme
from . import qe_lme_parallel
from .qe_fm import qelim_exists_lra_fm

qe_cooper = import_module("aria.quant.qe.qe_cooper")
qelim_exists_lia_cooper = qe_cooper.qelim_exists_lia_cooper


# Convenience imports
__all__ = [
    # Unified interface
    "ExternalQESolver",
    "QESolverConfig",
    "QEBackend",
    # Backward compatibility functions
    "eliminate_quantifiers_qepcad",
    "eliminate_quantifiers_mathematica",
    "eliminate_quantifiers_redlog",
    # Existing modules
    "qe_cooper",
    "qe_expansion",
    "qe_fm",
    "qe_lme",
    "qe_lme_parallel",
    "qelim_exists_lia_cooper",
    "qelim_exists_lra_fm",
]
