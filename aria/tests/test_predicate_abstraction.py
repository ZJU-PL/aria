"""Tests for the predicate-abstraction helpers."""

import z3

from aria.symabs.predicate_abstraction.predicate_abstraction import (
    eval_predicates,
    predicate_abstraction,
    prime_implicant,
)


def test_predicate_abstraction_returns_strongest_combination():
    i = z3.Int("i")
    formula = z3.And(i >= 0, i < 10, i + 1 >= 0)
    result = predicate_abstraction(formula, [i < 10, i >= 10])

    solver = z3.Solver()
    solver.add(result, z3.Not(i < 10))
    assert solver.check() == z3.unsat


def test_prime_implicant_returns_entailing_subset():
    x = z3.Int("x")
    formula = z3.And(x > 5, x < 10)
    predicates = [x > 0, x > 5, x < 10, x < 20]

    solver = z3.Solver()
    solver.add(formula)
    assert solver.check() == z3.sat
    literals = eval_predicates(solver.model(), predicates)

    core = prime_implicant(literals, formula)

    assert core
    entails = z3.Solver()
    entails.add(z3.And(*core), z3.Not(formula))
    assert entails.check() == z3.unsat


def test_prime_implicant_returns_empty_without_implicant():
    x = z3.Int("x")
    assert prime_implicant([x > 0], x > 10) == []
