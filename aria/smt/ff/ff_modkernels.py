#!/usr/bin/env python3
"""Prime-aware modular reduction kernels for finite-field translation.

This module classifies prime moduli and chooses reduction templates that can
be cheaper than generic ``x % p`` on structured fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import z3


@dataclass
class KernelSpec:
    """Selected modular-reduction kernel for one field modulus."""

    kind: str
    modulus: int
    k: int
    c: Optional[int] = None


class ModKernelSelector:
    """Classify field moduli and choose reduction kernels.

    Modes:
        auto: choose structured kernels when cheap and safe heuristics match.
        generic: always use plain modulo.
        structured: prefer structured kernels whenever classification matches.
    """

    def __init__(self, kernel_mode: str = "auto"):
        if kernel_mode not in ("auto", "generic", "structured"):
            raise ValueError("unknown kernel mode %s" % kernel_mode)
        self.kernel_mode = kernel_mode

    def classify(self, modulus: int) -> KernelSpec:
        """Return a kernel specification for *modulus*."""
        k = max(1, (modulus - 1).bit_length())
        if self.kernel_mode == "generic":
            return KernelSpec(kind="generic", modulus=modulus, k=k)

        if modulus <= 17:
            return KernelSpec(kind="small_prime_unrolled", modulus=modulus, k=k)

        pseudo = _detect_pseudo_mersenne(modulus)
        if pseudo is not None and (self.kernel_mode == "structured" or _is_low_cost(pseudo[1])):
            return KernelSpec(kind="pseudo_mersenne", modulus=modulus, k=pseudo[0], c=pseudo[1])

        near_sparse = _detect_near_power2_sparse(modulus)
        if near_sparse is not None and self.kernel_mode in ("auto", "structured"):
            return KernelSpec(
                kind="near_power2_sparse", modulus=modulus, k=near_sparse[0], c=near_sparse[1]
            )

        return KernelSpec(kind="generic", modulus=modulus, k=k)


class ModReducer:
    """Build z3 arithmetic terms for modular reduction according to kernel specs.

    All kernels are semantically equivalent to ``x % p`` for integer ``x``.
    Structured kernels still end with a final modulo to keep correctness
    straightforward and robust.
    """

    def __init__(self, specs: Dict[int, KernelSpec]):
        self.specs = specs

    def reduce(self, int_expr: z3.ArithRef, modulus: int) -> z3.ArithRef:
        """Return an Int expression equivalent to ``int_expr % modulus``."""
        spec = self.specs.get(modulus)
        if spec is None:
            return int_expr % modulus

        if spec.kind == "small_prime_unrolled":
            # For tiny primes, native modulo is already very competitive.
            return int_expr % modulus

        if spec.kind == "pseudo_mersenne" and spec.c is not None:
            # p = 2^k - c. Fold the high chunk into the low chunk.
            shift = 1 << spec.k
            folded = (int_expr % shift) + spec.c * (int_expr / shift)
            folded2 = (folded % shift) + spec.c * (folded / shift)
            return folded2 % modulus

        if spec.kind == "near_power2_sparse" and spec.c is not None:
            # p = 2^k + c. One fold step plus conservative final mod.
            shift = 1 << spec.k
            folded = (int_expr % shift) - spec.c * (int_expr / shift)
            return folded % modulus

        return int_expr % modulus


def _detect_pseudo_mersenne(modulus: int) -> Optional[tuple[int, int]]:
    """Match ``p = 2^k - c`` with small sparse ``c``."""
    k = modulus.bit_length()
    top = 1 << k
    c = top - modulus
    if c <= 0:
        return None
    if c.bit_length() <= 24 and _popcount(c) <= 4:
        return (k, c)
    return None


def _detect_near_power2_sparse(modulus: int) -> Optional[tuple[int, int]]:
    """Match ``p = 2^k + c`` where ``c`` is small and sparse."""
    k = modulus.bit_length() - 1
    if k <= 2:
        return None
    low = 1 << k
    c = modulus - low
    if c > 0 and c.bit_length() <= 20 and _popcount(c) <= 4:
        return (k, c)
    return None


def _is_low_cost(c: int) -> bool:
    return c <= (1 << 20) and _popcount(c) <= 3


def _popcount(value: int) -> int:
    return value.bit_count()
