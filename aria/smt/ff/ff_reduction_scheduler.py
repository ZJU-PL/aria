#!/usr/bin/env python3
"""Adaptive modulo-reduction scheduling for finite-field translations.

Scheduling policy determines when a translated arithmetic term is normalized
with ``mod p``. Earlier reduction shrinks terms but adds modulo operators;
later reduction reduces modulo pressure but can increase expression growth.
"""
from __future__ import annotations

from typing import Optional

from .ff_ast import FieldExpr
from .ff_ir import FFNodeStats

VALID_SCHEDULES = ("eager", "balanced", "lazy", "strict-recovery")


class ReductionScheduler:
    """Decide when to insert modulo reductions for field terms.

    Schedule tiers:
        eager: reduce after nearly every arithmetic step.
        balanced: reduce when growth/nonlinearity heuristics trigger.
        lazy: delay reductions aggressively.
        strict-recovery: conservative mode used for retry/fallback paths.
    """

    def __init__(self, schedule: str = "balanced"):
        if schedule not in VALID_SCHEDULES:
            raise ValueError("unknown reduction schedule %s" % schedule)
        self.schedule = schedule

    def should_reduce(
        self,
        expr: FieldExpr,
        modulus: int,
        stats: Optional[FFNodeStats],
        at_boundary: bool = False,
        before_kernel: bool = False,
    ) -> bool:
        """Return whether *expr* should be normalized modulo *modulus* now.

        Boundary and kernel-entry contexts always force a reduction to keep
        semantics explicit and to bound specialized-kernel inputs.
        """
        del expr

        if at_boundary or before_kernel:
            return True
        if self.schedule == "eager":
            return True

        if stats is None:
            return self.schedule != "lazy"

        field_bits = max(1, (modulus - 1).bit_length())

        if self.schedule == "strict-recovery":
            if stats.est_bits >= field_bits * 2:
                return True
            if stats.nonlinear:
                return True
            if stats.depth >= 8:
                return True
            return stats.fanout >= 2

        if self.schedule == "balanced":
            if stats.nonlinear and stats.fanout >= 2:
                return True
            if stats.est_bits >= field_bits * 3:
                return True
            if stats.depth >= 12 and stats.fanout >= 2:
                return True
            return False

        # lazy schedule: reduce only to avoid severe growth
        if stats.est_bits >= field_bits * 5:
            return True
        if stats.nonlinear and stats.fanout >= 3:
            return True
        return False


def stricter_schedule(schedule: str) -> Optional[str]:
    """Return the next stricter schedule, or None if already maximal."""
    order = ["lazy", "balanced", "strict-recovery", "eager"]
    if schedule not in order:
        return None
    idx = order.index(schedule)
    if idx >= len(order) - 1:
        return None
    return order[idx + 1]
