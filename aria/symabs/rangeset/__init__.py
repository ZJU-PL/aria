"""Range and Set Abstraction using SAT (after Barrett & King).

Public API
----------
- minimum(fml, x, signed=False): compute the minimal value of bit-vector ``x``
  subject to ``fml``.
- maximum(fml, x, signed=False): compute the maximal value of ``x``.
- range_abstraction(fml, x, signed=False): return (min_val, max_val).
- set_abstraction(fml, x, signed=False, steps=-1): return a list of inclusive
  intervals covering the feasible values of ``x``. If ``steps`` is negative,
  run to completion (exact). If positive and odd, result is an over-approx; if
  positive and even, an under-approx.

All APIs return Python integers under the chosen signedness, and also provide
bit-width-preserving Z3 bit-vector constants when needed internally.
"""

from .range_set_abstraction import (
    minimum,
    maximum,
    range_abstraction,
    set_abstraction,
)

__all__ = [
    "minimum",
    "maximum",
    "range_abstraction",
    "set_abstraction",
]
