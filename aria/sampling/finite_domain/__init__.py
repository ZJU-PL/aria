"""
Finite domain samplers.

This module provides samplers for finite domain formulas.

The samplers are organized by SMT theory:
- bool/: Boolean (SAT) samplers
- bv/: Bit-vector (QF_BV) samplers
- fp/: Floating-point (QF_FP) samplers
- uf/: Uninterpreted-function (QF_UF) samplers
- dt/: Algebraic datatype (QF_DT) samplers

All samplers implement the Sampler interface from aria.sampling.base and provide
a consistent API. Choose the appropriate sampler based on your logic and
sampling strategy requirements.
"""

from .bool import BooleanSampler
from .bv import BitVectorSampler, HashBasedBVSampler, QuickBVSampler
from .dt import DatatypeSampler
from .fp import FloatingPointSampler, HashBasedFPSampler, TotalOrderFPSampler
from .uf import UninterpretedFunctionSampler
from .ufdt import MixedUFDatatypeSampler

__all__ = [
    # Boolean samplers
    "BooleanSampler",
    # Bit-vector samplers
    "BitVectorSampler",  # Basic enumeration
    "HashBasedBVSampler",  # XOR-based uniform sampling
    "QuickBVSampler",  # QuickSampler for testing/fuzzing
    # Floating-point samplers
    "FloatingPointSampler",
    "HashBasedFPSampler",
    "TotalOrderFPSampler",
    # UF / datatype samplers
    "UninterpretedFunctionSampler",
    "DatatypeSampler",
    "MixedUFDatatypeSampler",
]
