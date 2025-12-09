"""
Random formula generation for testing purposes.

This module provides functionality for generating random logical formulas
and arithmetic expressions for testing and benchmarking purposes.
"""

from __future__ import annotations
from typing import List, Optional
from fractions import Fraction
import random

# Import from other SRK modules
from aria.srk.syntax import Context, Symbol, Expression, Formula, ArithTerm, Type, mk_var, mk_const, mk_real, mk_add, mk_mul, mk_leq, mk_and, mk_or, mk_exists, mk_forall


# Configuration variables (module-level, similar to OCaml refs)
min_coeff = -10
max_coeff = 10
quantifier_prefix = [
    'Forall', 'Forall', 'Forall', 'Forall',
    'Exists', 'Exists', 'Exists', 'Exists',
    'Forall', 'Forall', 'Forall', 'Forall'
]
formula_uq_proba = 0.9
formula_uq_depth = 5
number_of_monomials_per_expression = 5
dense = False


class RandomFormulaGenerator:
    """Generator for random formulas and terms."""

    def __init__(self, context: Context):
        self.context = context
        self.variable_counter = 0

    def mk_random_coeff(self) -> ArithTerm:
        """Generate a random coefficient."""
        coeff = random.randint(min_coeff, max_coeff)
        return mk_real(float(Fraction(coeff)))

    def mk_random_variable(self) -> ArithTerm:
        """Generate a random variable."""
        var_id = random.randint(0, len(quantifier_prefix) - 1)
        return mk_var(var_id, Type.INT)

    def mk_random_term(self) -> ArithTerm:
        """Generate a random arithmetic term."""
        if dense:
            # Dense representation - all variables
            num_vars = len(quantifier_prefix)
            terms = []
            for i in range(num_vars):
                coeff = self.mk_random_coeff()
                var = mk_var(i, Type.INT)
                terms.append(mk_mul([coeff, var]))
            return mk_add(terms)
        else:
            # Sparse representation - random subset of variables
            num_monomials = random.randint(1, number_of_monomials_per_expression)
            terms = []
            for _ in range(num_monomials):
                coeff = self.mk_random_coeff()
                var = self.mk_random_variable()
                terms.append(mk_mul([coeff, var]))
            return mk_add(terms)

    def mk_random_qf_formula(self, depth: int) -> Formula:
        """Generate a random quantifier-free formula."""
        if depth <= 0 or random.random() >= formula_uq_proba:
            # Generate a simple inequality
            left = self.mk_random_term()
            right = self.mk_random_coeff()
            return mk_leq(left, right)
        else:
            # Generate a compound formula
            left = self.mk_random_qf_formula(depth - 1)
            right = self.mk_random_qf_formula(depth - 1)

            if random.random() < 0.5:
                return mk_and([left, right])
            else:
                return mk_or([left, right])

    def mk_random_formula(self) -> Formula:
        """Generate a random quantified formula."""
        qf_formula = self.mk_random_qf_formula(formula_uq_depth)

        # Apply quantifiers from the prefix
        result = qf_formula

        for qt in reversed(quantifier_prefix):
            if qt == 'Exists':
                var_name = f"v{self.variable_counter}"
                self.variable_counter += 1
                result = mk_exists(var_name, Type.INT, result)
            elif qt == 'Forall':
                var_name = f"v{self.variable_counter}"
                self.variable_counter += 1
                result = mk_forall(var_name, Type.INT, result)

        return result

    def generate_formula(self) -> Formula:
        """Generate a random formula (convenience method)."""
        return self.mk_random_formula()

    def generate_term(self) -> ArithTerm:
        """Generate a random arithmetic term (convenience method)."""
        return self.mk_random_term()


# Convenience functions
def mk_random_formula(context: Context) -> Formula:
    """Generate a random formula in the given context."""
    generator = RandomFormulaGenerator(context)
    return generator.generate_formula()


def mk_random_term(context: Context) -> ArithTerm:
    """Generate a random arithmetic term in the given context."""
    generator = RandomFormulaGenerator(context)
    return generator.generate_term()


def mk_random_qf_formula(context: Context, depth: int = 5) -> Formula:
    """Generate a random quantifier-free formula."""
    generator = RandomFormulaGenerator(context)
    return generator.mk_random_qf_formula(depth)


# Configuration functions
def set_coeff_range(min_val: int, max_val: int) -> None:
    """Set the range for random coefficients."""
    RandomFormulaGenerator.min_coeff = min_val
    RandomFormulaGenerator.max_coeff = max_val


def set_quantifier_prefix(prefix: List[str]) -> None:
    """Set the quantifier prefix pattern."""
    RandomFormulaGenerator.quantifier_prefix = prefix


def set_formula_parameters(uq_proba: float = 0.9, uq_depth: int = 5, monomials: int = 5, dense: bool = False) -> None:
    """Set formula generation parameters."""
    RandomFormulaGenerator.formula_uq_proba = uq_proba
    RandomFormulaGenerator.formula_uq_depth = uq_depth
    RandomFormulaGenerator.number_of_monomials_per_expression = monomials
    RandomFormulaGenerator.dense = dense
