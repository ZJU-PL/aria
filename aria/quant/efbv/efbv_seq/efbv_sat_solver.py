import logging
# import sys
from pysat.formula import CNF
from pysat.solvers import Solver

logger = logging.getLogger(__name__)

"""
    cadical103  = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    cadical153  = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    gluecard3   = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    gluecard4   = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    glucose3    = ('g3', 'g30', 'glucose3', 'glucose30')
    glucose4    = ('g4', 'g41', 'glucose4', 'glucose41')
    lingeling   = ('lgl', 'lingeling')
    maplechrono = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    maplecm     = ('mcm', 'maplecm')
    maplesat    = ('mpl', 'maple', 'maplesat')
    mergesat3   = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    minicard    = ('mc', 'mcard', 'minicard')
    minisat22   = ('m22', 'msat22', 'minisat22')
    minisatgh   = ('mgh', 'msat-gh', 'minisat-gh')
"""
sat_solvers_in_pysat = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
                        'g4', 'lgl', 'mcb', 'mpl', 'mg3',
                        'mc', 'm22', 'mgh']


def solve_with_sat_solver(dimacs_str: str, solver_name: str) -> str:
    assert solver_name in sat_solvers_in_pysat
    # print(dimacs_str)
    print("Calling SAT solver {}".format(solver_name))
    pos = CNF(from_string=dimacs_str)
    # pos.to_fp(sys.stdout)
    aux = Solver(name=solver_name, bootstrap_with=pos)
    # print("solving via pysat")
    if aux.solve():
        return "sat"
    return "unsat"
