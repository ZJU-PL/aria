# coding: utf-8
from typing import List

from .clause import Clause
from .cnf import Cnf
from .variable import Variable

from pysat.formula import CNF


class PySATCNFReader:
    """
    Build an internal CNF object from PySAT CNF
    """

    def __init__(self) -> None:
        pass

    def read(self, cnf: CNF) -> Cnf:
        """
        Read PySAT CNF
        """
        clause_list: List[Clause] = []
        for cls in cnf.clauses:
            var_list: List[Variable] = []
            for var in cls:
                var_list.append(Variable(var))
            clause_list.append(Clause(var_list))
        cnf = Cnf(clause_list)
        return cnf


class NumericClausesReader:
    """
    Build an internal CNF object from Boolean clauses [[int]]
    """

    def __init__(self) -> None:
        pass

    def read(self, clauses: List[List[int]]) -> Cnf:
        """
        "Parse" numerical clauses
        """
        clause_list: List[Clause] = []
        for cls in clauses:
            var_list: List[Variable] = []
            for var in cls:
                if var != 0:
                    var_list.append(Variable(var))
            clause_list.append(Clause(var_list))
        cnf = Cnf(clause_list)
        return cnf
