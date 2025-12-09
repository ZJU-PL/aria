"""
The backbone module provides algorithms for computing backbones of 
Boolean formulas in both SAT and SMT contexts.

For SAT formulas, the backbone consists of literals that must be true
in all satisfying assignments of the formula.

For SMT formulas, the notion of backbone is less well-defined compared to SAT backbones.
"""

from .sat_backbone import (
    compute_backbone,
    compute_backbone_iterative,
    compute_backbone_chunking,
    compute_backbone_refinement,
    compute_backbone_with_approximation,
    is_backbone_literal,
    BackboneAlgorithm
)
