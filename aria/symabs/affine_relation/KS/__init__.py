"""Automatic Abstraction for Congruences (King & Søndergaard, VMCAI'10).

This lightweight package implements a practical variant of the paper's
congruent-closure based abstraction step and exposes a small API that integrates
with Z3. The focus is on bit-level congruences modulo ``m = 2**w`` over Boolean
variables, as in the original work.

Public API:
- ``CongruenceSystem``: data structure for a system of modular linear
  congruences A·x ≡ b (mod m)
- ``congruent_closure``: derive the strongest such system implied by a Boolean
  formula over a set of Boolean variables, by SAT-guided counterexample search
  (CEGIS-style).

The implementation aims to be simple and readable while remaining useful for
typical bit‑twiddling blocks. It can be strengthened later with triangular
matrix maintenance and other optimisations described in the paper.
"""

from .congruence_system import CongruenceSystem
from .congruence_abstraction import congruent_closure
from .loop_analysis import analyze_python_loop

__all__ = [
    "CongruenceSystem",
    "congruent_closure",
    "analyze_python_loop",
]
