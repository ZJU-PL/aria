"""
Finite domain samplers.

This module provides samplers for finite domain formulas (Boolean and bit-vector).

The samplers are organized by SMT theory:
- bool/: Boolean (SAT) samplers
- bv/: Bit-vector (QF_BV) samplers

All samplers implement the Sampler interface from aria.sampling.base and provide
a consistent API. Choose the appropriate sampler based on your logic and
sampling strategy requirements.
"""

from .bool import BooleanSampler
from .bv import BitVectorSampler, HashBasedBVSampler, QuickBVSampler

__all__ = [
    # Boolean samplers
    'BooleanSampler',

    # Bit-vector samplers
    'BitVectorSampler',      # Basic enumeration
    'HashBasedBVSampler',    # XOR-based uniform sampling
    'QuickBVSampler',        # QuickSampler for testing/fuzzing
]
