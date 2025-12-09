"""Abstract Domains of Affine Relations (Elder et al., SAS 11)

This package implements the abstract domains and algorithms described in
"Abstract Domains of Affine Relations" by Elder, Lim, Sharma, Andersen, and Reps.

The implementation includes:
- MOS (Müller-Olm/Seidl) domain: affine transformers as matrices
- KS (King/Søndergaard) domain: constraints on two-vocabulary relations
- AG (Affine Generator) domain: generators with diagonal decomposition
- Conversion algorithms between domains
- Symbolic α function implementation
- Interprocedural analysis support

Public API:
- `MOS`: MOS domain elements and operations
- `KS`: KS domain elements and operations
- `AG`: AG domain elements and operations
- `howellize`: Convert matrix to Howell form
- `make_explicit`: Convert AG matrix to near-explicit form
- `alpha_mos`: Symbolic implementation of α function for MOS
- `alpha_ks`: Symbolic implementation of α function for KS
- `alpha_ag`: Symbolic implementation of α function for AG
"""

from .mos_domain import MOS, alpha_mos, create_z3_variables
from .ks_domain import KS, alpha_ks
from .ag_domain import AG, alpha_ag
from .matrix_ops import howellize, make_explicit
from .conversions import mos_to_ks, ks_to_mos, ag_to_ks, ks_to_ag
__all__ = [
    "MOS",
    "KS",
    "AG",
    "howellize",
    "make_explicit",
    "alpha_mos",
    "alpha_ks",
    "alpha_ag",
    "create_z3_variables",
    "mos_to_ks",
    "ks_to_mos",
    "ag_to_ks",
    "ks_to_ag",
]
