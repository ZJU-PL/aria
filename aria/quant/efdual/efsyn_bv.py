"""Bit-vector exists-forall synthesis procedure.

Compared to efsyn_simple.py:

It makes the universal-side memory more reusable by storing **bit-slice cubes** instead of only raw
counterexamples. A failed $Y$ gets generalized into guards like:

$$\text{Extract}(7, 4, y) = c_1 \;\wedge\; \text{Extract}(3, 0, y) = c_2$$

That tends to work much better than keeping only full concrete words when attacks cluster by shared
high/low nibble patterns.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from z3 import *

try:
    from .efsyn_common import (
        Assignment,
        Attack,
        Candidate,
        DualMemoryCEGISBase,
        expr_mentions_any,
        is_bv_sort,
    )
except ImportError:  # pragma: no cover - direct script/module execution fallback
    from efsyn_common import (
        Assignment,
        Attack,
        Candidate,
        DualMemoryCEGISBase,
        expr_mentions_any,
        is_bv_sort,
    )


# =============================================================================
# 1) Bit-vector specialized version
# =============================================================================

class BVExistsForallCEGIS(DualMemoryCEGISBase):
    """
    Bit-vector specialization.

    Main specialization:
      - exact checks use SolverFor("QF_BV")
      - counterexamples generalize to bit-slice cubes:
            Extract(hi, lo, y) == const
      - novelty is also slice-aware
    """

    def __init__(
        self,
        *,
        slice_width: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.slice_width = max(1, int(slice_width))

        for v in self.x_vars + self.y_vars:
            if not is_bv_sort(v.sort()):
                raise TypeError(
                    "BVExistsForallCEGIS requires all X and Y variables to be BitVec sort."
                )

    def _new_solver(self) -> Solver:
        s = SolverFor("QF_BV")
        s.set(timeout=self.timeout_ms)
        return s

    def _bit_slices(self, bv: BitVecRef) -> List[Tuple[int, int]]:
        w = bv.size()
        out: List[Tuple[int, int]] = []
        lo = 0
        while lo < w:
            hi = min(w - 1, lo + self.slice_width - 1)
            out.append((hi, lo))
            lo += self.slice_width
        return out

    def _add_y_novelty_soft(self, opt: Optimize, recent: Sequence[Attack]) -> None:
        for atk in recent:
            for yv, val in zip(self.y_vars, atk.rep):
                opt.add_soft(yv != val, weight="1", id="novel-y")
                for hi, lo in self._bit_slices(yv):
                    opt.add_soft(
                        Extract(hi, lo, yv) != simplify(Extract(hi, lo, val)),
                        weight="1",
                        id="novel-y-slice",
                    )

    def _basis_for_generalization(
        self,
        cand: Candidate,
        y_vals: Assignment,
        fail_formula: BoolRef,
    ) -> List[BoolRef]:
        basis: List[BoolRef] = []

        # Fine-grained slice equalities.
        for yv, yval in zip(self.y_vars, y_vals):
            basis.append(yv == yval)
            for hi, lo in self._bit_slices(yv):
                basis.append(Extract(hi, lo, yv) == simplify(Extract(hi, lo, yval)))

        # If user also supplied a basis, include its instantiated y-only atoms.
        if self.region_basis:
            for atom in self.region_basis:
                inst = simplify(self._instantiate(atom, self.x_vars, cand.vals))
                if (
                    is_bool(inst)
                    and not expr_mentions_any(inst, self.x_vars)
                    and expr_mentions_any(inst, self.y_vars)
                ):
                    basis.append(inst)

        # Dedup.
        seen = set()
        out: List[BoolRef] = []
        for b in basis:
            key = simplify(b).sexpr()
            if key not in seen:
                out.append(simplify(b))
                seen.add(key)
            if len(out) >= self.max_region_atoms:
                break
        return out

if __name__ == "__main__":
    print("=== Bit-vector example ===")
    # Unsigned example:
    #   exists x in BV8 .
    #   forall y in [0_u, 13_u] . y <=_u x
    # Smallest valid witness is x = 13.

    x = BitVec("x", 8)
    y = BitVec("y", 8)

    bv_solver = BVExistsForallCEGIS(
        x_vars=[x],
        y_vars=[y],
        predicate=ULE(y, x),
        domain_x=BoolVal(True),
        domain_y=ULE(y, BitVecVal(13, 8)),
        slice_width=4,
        max_iters=20,
        max_x_memory=16,
        max_y_memory=16,
        verbose=True,
    )

    bv_result = bv_solver.solve()
    print("BV status :", bv_result.status)
    print("BV witness:", bv_result.witness)
    print("BV iters  :", bv_result.iterations)
    print("BV msg    :", bv_result.message)
    print()
