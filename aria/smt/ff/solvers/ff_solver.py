#!/usr/bin/env python3
"""
ff_solver.py  –  Front-end entry points and backend selection for finite fields.
"""
from __future__ import annotations

from typing import Optional

import z3

from ..core.ff_ast import ParsedFormula
from .ff_bv_solver import FFBVSolver
from .ff_bv_solver2 import FFBVBridgeSolver
from .ff_int_solver import FFIntSolver
from .ff_perf_solver import FFPerfSolver


class FFAutoSolver:
    """Choose a backend based on the field sizes present in the formula.

    Backend routing:
    - small fields: bit-vector backend
    - medium fields: BV/Int bridge backend
    - large fields: performance backend by default (can be disabled)
    """

    def __init__(
        self,
        small_bv_bits: int = 31,
        medium_bridge_bits: int = 160,
        enable_perf_backend: bool = True,
        perf_policy: str = "auto",
    ):
        """Create an automatic finite-field solver.

        Args:
            small_bv_bits: Max bit-width for direct BV backend.
            medium_bridge_bits: Max bit-width for BV/Int bridge backend.
            enable_perf_backend: Allow routing to FFPerfSolver.
            perf_policy: ``auto``/``always``/``never``/``large-prime``.
        """
        self.small_bv_bits = small_bv_bits
        self.medium_bridge_bits = medium_bridge_bits
        self.enable_perf_backend = enable_perf_backend
        self.perf_policy = perf_policy
        self.backend_name: Optional[str] = None
        self.backend = None

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Select a backend and solve the formula."""
        self.backend_name = self._select_backend(formula)
        if self.backend_name == "bv":
            self.backend = FFBVSolver()
        elif self.backend_name == "bv2":
            self.backend = FFBVBridgeSolver()
        elif self.backend_name == "perf":
            self.backend = FFPerfSolver()
        else:
            self.backend = FFIntSolver()
        return self.backend.check(formula)

    def model(self) -> Optional[z3.ModelRef]:
        """Return the backend model when available."""
        if self.backend is None:
            return None
        return self.backend.model()

    def _select_backend(self, formula: ParsedFormula) -> str:
        if not formula.field_sizes:
            return "int"
        max_bits = max((modulus - 1).bit_length() for modulus in formula.field_sizes)
        if self._use_perf_backend(max_bits):
            return "perf"
        if max_bits <= self.small_bv_bits:
            return "bv"
        if max_bits <= self.medium_bridge_bits:
            return "bv2"
        return "int"

    def _use_perf_backend(self, max_bits: int) -> bool:
        if not self.enable_perf_backend:
            return False
        if self.perf_policy == "always":
            return True
        if self.perf_policy in ("auto", "large-prime"):
            return max_bits > self.medium_bridge_bits
        if self.perf_policy == "never":
            return False
        raise ValueError("unknown perf_policy %s" % self.perf_policy)
