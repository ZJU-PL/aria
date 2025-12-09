# coding: utf-8

"""
Conjunctive Normal Form expression simplifier that preserves satisfiability
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cnf import Cnf


def cnf_tautoly_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are tautology
    :complexity: O(c)
    """
    cnf = cnf.tautology_elimination()
    return cnf


def cnf_blocked_clause_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are blocked
    :complexity: O( (c*l)^2 )
    """
    cnf = cnf.blocked_clause_elimination()
    return cnf


def cnf_subsumption_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are subsumed
    :complexity: O(  )
    """
    cnf = cnf.subsumption_elimination()
    return cnf


def cnf_hidden_tautoly_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are hidden tautology
    :complexity: O( (c*l)^2 )
    """
    cnf = cnf.hidden_tautology_elimination()
    return cnf


def cnf_hidden_blocked_clause_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are hidden blocked
    :complexity: O( (c*l)^2 )
    """
    cnf = cnf.hidden_blocked_clause_elimination()
    return cnf


def cnf_hidden_subsumption_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are hidden subsumed
    :complexity: O( (l*c)^2 )
    """
    cnf = cnf.hidden_subsumption_elimination()
    return cnf


def cnf_asymmetric_tautoly_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are asymmetric tautology
    :complexity: O( c^2 * l^2 * 2^l )
    """
    cnf = cnf.asymmetric_tautology_elimination()
    return cnf


def cnf_asymmetric_blocked_clause_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are asymmetric blocked
    :complexity: O( c^2 * l^2 * 2^l )
    """
    cnf = cnf.asymmetric_blocked_clause_elimination()
    return cnf


def cnf_asymmetric_subsumption_elimination(cnf: 'Cnf') -> 'Cnf':
    """
    Simplify CNF by removing all clauses that are asymmetric subsumed
    :complexity: O( c^2 * l^2 * 2^l )
    """
    cnf = cnf.asymmetric_subsumption_elimination()
    return cnf


def explicits(cnf: 'Cnf') -> 'Cnf':
    """
    The explicits function takes a CNF object as an argument and returns the same CNF with
    the following transformations applied:

    :param cnf: Store the cnf formula
    :return: The cnf after performing all the functions
    """
    cnf = cnf.subsumption_elimination()
    cnf = cnf.blocked_clause_elimination()
    cnf = cnf.tautology_elimination()
    return cnf


def hiddens(cnf: 'Cnf') -> 'Cnf':
    """
    The hiddens function takes a CNF object as input and returns a new CNF object
    with the following transformations applied:

    :param cnf: Store the cnf formula
    :return: A new cnf object that has been optimized by the 3 hidden elimination methods
    """
    cnf = cnf.hidden_subsumption_elimination()
    cnf = cnf.hidden_tautology_elimination()
    cnf = cnf.hidden_blocked_clause_elimination()
    return cnf


def asymmetrics(cnf: 'Cnf') -> 'Cnf':
    """
    :param cnf: Store the cnf formula
    :return: A cnf object
    """
    cnf = cnf.asymmetric_subsumption_elimination()
    cnf = cnf.asymmetric_tautology_elimination()
    cnf = cnf.asymmetric_blocked_clause_elimination()
    return cnf


def complete(cnf: 'Cnf') -> 'Cnf':
    """Use at your risk"""
    cnf = cnf.asymmetric_subsumption_elimination()

    cnf = cnf.blocked_clause_elimination()
    cnf = cnf.tautology_elimination()
    cnf = cnf.subsumption_elimination()

    # cnf = cnf.hidden_tautology_elimination()
    cnf = cnf.asymmetric_tautology_elimination()

    # cnf = cnf.hidden_blocked_clause_elimination()
    cnf = cnf.asymmetric_blocked_clause_elimination()

    # cnf = cnf.hidden_subsumption_elimination()
    # cnf = cnf.asymmetric_subsumption_elimination()

    return cnf
