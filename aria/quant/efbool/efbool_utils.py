from enum import Enum, auto

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
sat_solvers = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
               'g4', 'lgl', 'mcb', 'mpl', 'mg3',
               'mc', 'm22', 'msh']


class SATSolver(Enum):
    """Enumeration of SAT solvers and their aliases."""
    CADICAL103 = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    CADICAL153 = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    GLUECARD3 = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    GLUECARD4 = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    GLUCOSE3 = ('g3', 'g30', 'glucose3', 'glucose30')
    GLUCOSE4 = ('g4', 'g41', 'glucose4', 'glucose41')
    LINGELING = ('lgl', 'lingeling')
    MAPLECHRONO = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    MAPLECM = ('mcm', 'maplecm')
    MAPLESAT = ('mpl', 'maple', 'maplesat')
    MERGESAT3 = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    MINICARD = ('mc', 'mcard', 'minicard')
    MINISAT22 = ('m22', 'msat22', 'minisat22')
    MINISATGH = ('mgh', 'msat-gh', 'minisat-gh')

    @classmethod
    def get_solver_names(cls) -> list[str]:
        """Get a list of all solver names and aliases."""
        return [alias for solver in cls for alias in solver.value]


class EFBoolResult(Enum):
    """Result of EFBool Checking"""
    UNSAT = auto()
    SAT = auto()
    UNKNOWN = auto()
    ERROR = auto()
