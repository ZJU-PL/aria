#!/usr/bin/env python3
"""
ff_numbertheory.py  –  Small arithmetic helpers for finite-field front-ends.
"""
from __future__ import annotations


_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def is_probable_prime(n: int) -> bool:
    """Return whether *n* is prime.

    For machine-sized integers this is deterministic. For larger integers the
    fixed-base Miller-Rabin test used here is still exact for the benchmark
    moduli in this repository and fast enough for solver setup.
    """
    if n < 2:
        return False
    for prime in _SMALL_PRIMES:
        if n == prime:
            return True
        if n % prime == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    if n < 341550071728321:
        bases = [2, 3, 5, 7, 11, 13, 17]
    else:
        bases = _SMALL_PRIMES

    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        witness = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                witness = False
                break
        if witness:
            return False
    return True
