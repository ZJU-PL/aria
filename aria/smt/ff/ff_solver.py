#!/usr/bin/env python3
"""
ff_solver.py  –  Front-end entry points and backend selection for finite fields.
"""
from __future__ import annotations

from typing import Optional

import z3

from .ff_ast import ParsedFormula
from .ff_bv_solver import FFBVSolver
from .ff_bv_solver2 import FFBVBridgeSolver
from .ff_int_solver import FFIntSolver


class FFAutoSolver:
    """Choose a backend based on the field sizes present in the formula."""

    def __init__(self, small_bv_bits: int = 31, medium_bridge_bits: int = 160):
        self.small_bv_bits = small_bv_bits
        self.medium_bridge_bits = medium_bridge_bits
        self.backend_name: Optional[str] = None
        self.backend = None

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Select a backend and solve the formula."""
        self.backend_name = self._select_backend(formula)
        if self.backend_name == "bv":
            self.backend = FFBVSolver()
        elif self.backend_name == "bv2":
            self.backend = FFBVBridgeSolver()
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
        if max_bits <= self.small_bv_bits:
            return "bv"
        if max_bits <= self.medium_bridge_bits:
            return "bv2"
        return "int"
