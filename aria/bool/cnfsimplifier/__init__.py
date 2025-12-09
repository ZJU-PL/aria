from typing import List

from .io import NumericClausesReader, PySATCNFReader
from .simplifier import cnf_subsumption_elimination, cnf_hidden_subsumption_elimination, \
    cnf_asymmetric_subsumption_elimination, cnf_asymmetric_tautoly_elimination, \
    cnf_tautoly_elimination, cnf_hidden_tautoly_elimination, cnf_blocked_clause_elimination, \
    cnf_hidden_blocked_clause_elimination


def simplify_numeric_clauses(clauses: List[List[int]]) -> List[List[int]]:
    """
    :param clauses: numerical clauses
    :return: simplified clauses
    """
    cnf = NumericClausesReader().read(clauses)
    new_cnf = cnf_subsumption_elimination(cnf)  # why only subsumption?
    return new_cnf.get_numeric_clauses()
