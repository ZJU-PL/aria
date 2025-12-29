"""
Quantifier Elimination (QE) module for aria.

This module provides various approaches to quantifier elimination:
- External tools: QEPCAD, Mathematica, Redlog
- Internal algorithms: Shannon expansion, LME-based methods
- Unified interface for external tools
"""

# Import the unified external QE solver
from .external_qe import (
    ExternalQESolver,
    QESolverConfig,
    QEBackend,
    eliminate_quantifiers_qepcad,
    eliminate_quantifiers_mathematica,
    eliminate_quantifiers_redlog
)

# Import existing modules for backward compatibility
from . import qe_expansion
from . import qe_lme
from . import qe_lme_parallel


# Convenience imports
__all__ = [
    # Unified interface
    'ExternalQESolver',
    'QESolverConfig',
    'QEBackend',

    # Backward compatibility functions
    'eliminate_quantifiers_qepcad',
    'eliminate_quantifiers_mathematica',
    'eliminate_quantifiers_redlog',

    # Existing modules
    'qe_expansion',
    'qe_lme',
    'qe_lme_parallel',
]
