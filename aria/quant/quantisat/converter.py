"""Converter for transforming quantifier expressions to SMT2 format."""
from __future__ import annotations

from typing import Dict, List, Tuple

import sympy as sp

from aria.quant.quantisat.expression import (
    get_function_expression, get_polynomial_expression)
from aria.quant.quantisat.quantifier import Exists, ForAll, Quantifier
from aria.quant.quantisat.util import to_smt


def convert(constraint_system: List[Quantifier], degree: int) -> str:
    """
    Convert a quantified formula to an SMT2 formula.

    Parameters
    ----------
    constraint_system : List[Quantifier]
        The quantified formula to be converted.
    degree : int
        The degree of the polynomial templates.

    Returns
    -------
    str
        The SMT2 formula.
    """
    converter = Converter(degree)
    c = converter.convert(constraint_system)
    return c


class Converter:
    """
    The converter class to convert a quantified formula to an SMT2 formula.
    """

    def __init__(self, degree: int):
        """
        Initialize the converter.

        Parameters
        ----------
        degree : int
            The degree of the polynomial templates.
        """
        self.degree = degree

    def convert(self, constraint_system: List[Quantifier]) -> str:
        """
        Convert a quantified formula to an SMT2 formula.

        Parameters
        ----------
        constraint_system : List[Quantifier]
            The system of quantified formulas to be converted.

        Returns
        -------
        str
            The SMT2 formula.
        """
        forall_quant_vars = set()
        exists_quant_vars = {}
        ground_formulas = []
        assertions = []

        for quantified_formula in constraint_system:
            (this_forall_quant_vars, this_exists_quant_vars,
             this_ground_formula, this_assertion) = self.extract(
                quantified_formula)

            forall_quant_vars.update(this_forall_quant_vars)

            for exists_var, preceding_vars in this_exists_quant_vars.items():
                if exists_var in exists_quant_vars:
                    exists_quant_vars[exists_var].update(preceding_vars)
                else:
                    exists_quant_vars[exists_var] = set(preceding_vars)
            ground_formulas.append(this_ground_formula)
            assertions.append(this_assertion)

        if self.degree is not None:
            transform = {
                var: get_polynomial_expression(
                    f"a_{i}", preceding_vars, self.degree)
                for i, (var, preceding_vars)
                in enumerate(exists_quant_vars.items())}
        else:
            transform = {
                var: get_function_expression(var, preceding_vars)
                for var, preceding_vars in exists_quant_vars.items()}

        for i, this_ground_formula in enumerate(ground_formulas):

            this_ground_formula = this_ground_formula.subs(transform)
            ground_formulas[i] = this_ground_formula
            assertions[i] = assertions[i].subs(transform)

        smt2 = self.generate_smt2(exists_quant_vars, list(
            forall_quant_vars), assertions, ground_formulas)

        return smt2

    def generate_smt2(
            self, exists_quant_vars: Dict[sp.Symbol, List[sp.Symbol]],
            forall_quant_vars: List[sp.Symbol],
            assertions: List[sp.Basic],
            ground_formulas: List[sp.Basic]) -> str:
        """
        Generate an SMT2 formula from the extracted information of the quantified formulas.

        Parameters
        ----------
        exists_quant_vars : Dict[sp.Symbol, List[sp.Symbol]]
            The exists-quantified variables and their forall-predecessors.
        forall_quant_vars : List[sp.Symbol]
            The forall-quantified variables.
        assertions : List[sp.Basic]
            The assertions to be added to the left side of the constraint pairs.
        ground_formulas : List[sp.Basic]
            The formulas without quantifiers.

        Returns
        -------
        str
            The SMT2 formula.
        """
        smt_lines = []

        if self.degree is not None:
            free_vars = set()
            for assertion, ground_formula in zip(assertions, ground_formulas):
                free_vars.update(assertion.free_symbols)
                free_vars.update(ground_formula.free_symbols)
            free_vars = free_vars - set(forall_quant_vars)
            for var in free_vars:
                smt_lines.append(f'(declare-const {var.name} Real)')
        else:
            for exists_var, predecessor in exists_quant_vars.items():
                real_types = " ".join(["Real"] * len(predecessor))
                smt_lines.append(
                    f'(declare-fun {exists_var.name} ({real_types}) Real)')

        forall_vars = [f'({var.name} Real)' for var in forall_quant_vars]
        forall_vars = f"({' '.join(forall_vars)})"

        for assertion, ground_formula in zip(assertions, ground_formulas):
            if len(forall_quant_vars) == 0:
                smt_lines.append(
                    f'(assert (=> {to_smt(assertion)} {to_smt(ground_formula)}))')
            else:
                smt_lines.append(
                    f'(assert (forall {forall_vars} (=> {to_smt(assertion)} {to_smt(ground_formula)})))')

        smt_lines.append('(check-sat)')
        smt_lines.append('(get-model)')

        return '\n'.join(smt_lines)

    def extract(
            self, quantified_formula: Quantifier
    ) -> Tuple[List[sp.Symbol], Dict[sp.Symbol, List[sp.Symbol]],
               sp.Basic, sp.Basic]:
        """
        Extract information from a quantified formula such that we can construct constraints from it.

        Parameters
        ----------
        quantified_formula : Quantifier
            The quantified formula to extract the variables and substitutions.

        Returns
        -------
        forall_quant_vars : List[sp.Symbol]
            The list of forall-quantified variables.
        exists_quant_vars : Dict[sp.Symbol, List[sp.Symbol]]
            The exists-quantified variables and their forall-predecessors.
        ground_formula : sp.Basic
            The formula without quantifiers.
        assertion : sp.Basic
            Assertions to be added to the left side of the constraint pairs.
        """
        forall_quant_vars = []
        exists_quant_vars = {}

        while isinstance(quantified_formula, Quantifier):
            if isinstance(quantified_formula, ForAll):
                forall_quant_vars.extend(quantified_formula.variables)
            elif isinstance(quantified_formula, Exists):
                for var in quantified_formula.variables:
                    exists_quant_vars[var] = forall_quant_vars.copy()
            quantified_formula = quantified_formula.formula

        ground_formula = quantified_formula
        if isinstance(ground_formula, sp.Implies):
            assertion = ground_formula.args[0]
            ground_formula = ground_formula.args[1]
        else:
            assertion = sp.true

        return forall_quant_vars, exists_quant_vars, ground_formula, assertion
