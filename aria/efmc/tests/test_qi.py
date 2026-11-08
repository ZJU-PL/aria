"""Tests for the quantifier-instantiation verification path."""

import z3

from aria.efmc.engines.qi import QuantifierInstantiationProver
from aria.efmc.sts import TransitionSystem


def _counter_system(expected):
    x = z3.Int("x")
    x_prime = z3.Int("x!")
    post = x >= 0 if expected == "safe" else x < 0
    return TransitionSystem(
        variables=[x],
        prime_variables=[x_prime],
        init=(x == 0),
        trans=(x_prime == x + 1),
        post=post,
    )


def test_qi_proves_safe_system():
    prover = QuantifierInstantiationProver(
        _counter_system("safe"), qi_strategy="mbqi", solver="z3api"
    )

    result = prover.solve(timeout=10)

    assert result.is_safe
    assert not result.is_unsafe


def test_qi_finds_no_invariant_for_unsafe_system():
    prover = QuantifierInstantiationProver(
        _counter_system("unsafe"), qi_strategy="mbqi", solver="z3api"
    )

    result = prover.solve(timeout=10)

    assert result.is_unsafe
    assert not result.is_safe
