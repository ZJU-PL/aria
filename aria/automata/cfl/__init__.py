"""Context-free language (CFL) reachability package."""

from aria.automata.cfl.cfl_solver import CFLSolver
from aria.automata.cfl.pag_matrix import PAG_Matrix
from aria.automata.cfl.sc_solver import SCSolver

# Backward compatibility aliases
PAGMatrix = PAG_Matrix
SCReducer = SCSolver

__all__ = [
    "CFLSolver",
    "PAGMatrix",
    "PAG_Matrix",
    "SCReducer",
    "SCSolver",
]
