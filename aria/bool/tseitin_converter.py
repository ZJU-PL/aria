# coding: utf-8
"""
Boolean Formula Conversion Utilities

This module provides facilities for converting between different forms of Boolean formulas,
with a focus on Tseitin's transformation algorithm. The transformation converts a formula
in Disjunctive Normal Form (DNF) to an equisatisfiable formula in Conjunctive Normal Form (CNF).

Key Features:
- DNF to CNF conversion using Tseitin's transformation
- Preserves satisfiability while potentially introducing auxiliary variables
- Produces a formula that is linear in the size of the input
"""
from typing import List, Set
from dataclasses import dataclass


@dataclass
class TseitinResult:
    """Contains the result of Tseitin transformation and metadata"""
    cnf: List[List[int]]
    auxiliary_vars: Set[int]
    original_vars: Set[int]


def get_variable_range(formula: List[List[int]]) -> int:
    """
    Find the maximum absolute value of variables in the formula.
    This helps in determining where to start numbering auxiliary variables.

    Args:
        formula: A nested list of integers representing logical clauses

    Returns:
        The maximum absolute value found in the formula

    Raises:
        ValueError: If the formula is empty or contains invalid variables
    """
    if not formula or not all(clause for clause in formula):
        raise ValueError("Formula cannot be empty or contain empty clauses")

    return max(abs(var) for clause in formula for var in clause)


def tseitin(dnf: List[List[int]]) -> TseitinResult:
    """
    Implements Tseitin's transformation to convert DNF to CNF.

    The transformation preserves satisfiability by introducing auxiliary variables.
    For each clause in the DNF, it creates an auxiliary variable and adds clauses
    to maintain logical equivalence.

    Args:
        dnf: A nested list of integers representing the DNF formula.
             Each inner list represents a conjunction of literals.
             Positive integers represent positive literals,
             negative integers represent negated literals.

    Returns:
        TseitinResult containing:
        - The resulting CNF formula
        - Set of auxiliary variables introduced
        - Set of original variables

    Raises:
        ValueError: If input DNF is empty or invalid
    """
    if not dnf:
        raise ValueError("DNF formula cannot be empty")

    # Collect original variables and verify input
    original_vars = {abs(var) for clause in dnf for var in clause}
    next_aux_var = get_variable_range(dnf) + 1
    auxiliary_vars = set()
    cnf_clauses = []

    # Process each DNF clause
    for clause in dnf:
        # Create auxiliary variable for current clause
        aux_var = next_aux_var
        auxiliary_vars.add(aux_var)

        # Add clause connecting auxiliary variable to original clause
        cnf_clauses.append([-1 * var for var in clause] + [aux_var])

        # Add implications for each literal in the clause
        for literal in clause:
            cnf_clauses.append([literal, -1 * aux_var])

        next_aux_var += 1

    return TseitinResult(
        cnf=cnf_clauses,
        auxiliary_vars=auxiliary_vars,
        original_vars=original_vars
    )


def format_formula(formula: List[List[int]], form_type: str = "CNF") -> str:
    """
    Creates a human-readable string representation of a Boolean formula.

    Args:
        formula: The Boolean formula to format
        form_type: The type of formula ("CNF" or "DNF")

    Returns:
        A string representation of the formula
    """
    clause_op = " ∧ " if form_type == "CNF" else " ∨ "
    return clause_op.join(
        f"({' ∨ ' if form_type == 'CNF' else ' ∧ '}"
        f"{' '.join([str(lit) for lit in clause])})"
        for clause in formula
    )


def run_tests():
    """Comprehensive test cases for the Tseitin transformation"""
    # Test case 1: Simple DNF
    dnf1 = [[-1, -2, 4], [1, -4]]
    result1 = tseitin(dnf1)
    print(f"Original DNF: {format_formula(dnf1, 'DNF')}")
    print(f"Resulting CNF: {format_formula(result1.cnf, 'CNF')}")
    print(f"Auxiliary variables: {result1.auxiliary_vars}")
    print(f"Original variables: {result1.original_vars}\n")

    # Test case 2: More complex DNF
    dnf2 = [[1, 2, 3], [-1, -3], [2, -4, 5]]
    result2 = tseitin(dnf2)
    print(f"Original DNF: {format_formula(dnf2, 'DNF')}")
    print(f"Resulting CNF: {format_formula(result2.cnf, 'CNF')}")
    print(f"Auxiliary variables: {result2.auxiliary_vars}")
    print(f"Original variables: {result2.original_vars}")


if __name__ == "__main__":
    run_tests()
