"""
SAE Unknown Formula Retriever

Uses Skeletal Approximation Enumeration (SAE) and structural mutations to resolve unknown SMT formulas.
"""

from .resolver import SAEUnknownResolver

__version__ = "1.0.0"
__all__ = ["SAEUnknownResolver"]