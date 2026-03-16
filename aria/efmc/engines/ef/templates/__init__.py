# coding: utf-8
"""
Template-based abstract domains for program verification.

This module provides various abstract domains for template-based program analysis,
including numeric (interval, zone, octagon, affine, polyhedron) and
bit-vector templates, as well as disjunctive variants.
"""

# Base classes
from .abstract_template import TemplateType, Template

# Numeric templates
from .interval import IntervalTemplate, DisjunctiveIntervalTemplate
from .intervalv2 import IntervalTemplateV2
from .zone import ZoneTemplate
from .octagon import OctagonTemplate
from .affine import AffineTemplate, DisjunctiveAffineTemplate
from .polyhedron import PolyTemplate, DisjunctivePolyTemplate

# Bit-vector templates
from .bv_interval import BitVecIntervalTemplate, DisjunctiveBitVecIntervalTemplate
from .bv_zone import BitVecZoneTemplate, DisjunctiveBitVecZoneTemplate
from .bv_octagon import BitVecOctagonTemplate, DisjunctiveBitVecOctagonTemplate
from .bv_affine import BitVecAffineTemplate, DisjunctiveBitVecAffineTemplate
from .bv_polyhedron import BitVecPolyhedronTemplate, DisjunctiveBitVecPolyhedronTemplate
from .bv_bitwise import KnownBitsTemplate, BitPredAbsTemplate

# Bit-level templates
from .bv_pattern import BitVecPatternTemplate, DisjunctiveBitVecPatternTemplate
from .bv_rotation import BitVecRotationTemplate, DisjunctiveBitVecRotationTemplate
from .bv_xor_parity import BitVecXorParityTemplate, DisjunctiveBitVecXorParityTemplate

# Bit-vector ranking function templates for termination verification
from .bv_ranking import (
    BitVecLinearRankingTemplate,
    BitVecLexicographicRankingTemplate,
    BitVecConditionalRankingTemplate,
)

# Bit-vector recurrence set templates for non-termination verification
from .bv_recurrence import (
    BitVecLinearRecurrenceTemplate,
    BitVecIntervalRecurrenceTemplate,
    BitVecDisjunctiveRecurrenceTemplate,
)

# Hybrid bit-vector templates for combining multiple domains
from .bv_hybrid import ConjunctiveBitVecTemplate

# Floating-point templates for QF_FP support
from .fp_interval import FPIntervalTemplate
from .fp_polyhedron import FPPolyhedronTemplate

# No need to redefine the classes with the same name - they're already available
# in the namespace after the imports above
