"""Backbone utilities across Boolean and SMT reasoning.

SAT backbone functionality now lives under ``aria.bool.backbone`` and is
re-exported here for compatibility. SMT-specific helpers remain local to this
package.
"""

from .sat_backbone import (
    BackboneAlgorithm,
    compute_backbone,
    compute_backbone_chunking,
    compute_backbone_iterative,
    compute_backbone_refinement,
    compute_backbone_with_approximation,
    is_backbone_literal,
)
