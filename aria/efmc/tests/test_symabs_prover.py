"""Tests for the EFMC symbolic-abstraction prover."""

import z3

from aria.efmc.engines.symabs import SymbolicAbstractionProver
from aria.efmc.sts import TransitionSystem


def test_symbolic_abstraction_prover_handles_integer_interval_domain():
    x = z3.Int("x")
    x_prime = z3.Int("x!")
    sts = TransitionSystem(
        variables=[x],
        prime_variables=[x_prime],
        init=(x == 0),
        trans=(x_prime == x + 1),
        post=(x >= 0),
    )

    prover = SymbolicAbstractionProver(sts)
    prover.domain = "interval"
    result = prover.solve()

    assert result.is_safe


def test_symbolic_abstraction_prover_handles_bitvector_interval_domain():
    x = z3.BitVec("x", 8)
    x_prime = z3.BitVec("x!", 8)
    sts = TransitionSystem(
        variables=[x],
        prime_variables=[x_prime],
        init=(x == 0),
        trans=(x_prime == x),
        post=(x == 0),
    )

    prover = SymbolicAbstractionProver(sts)
    prover.domain = "interval"
    result = prover.solve()

    assert result.is_safe
