"""
Format translators and converters for SMT, SAT, and optimization problems.

This package provides converters between various constraint/solver formats:

CNF/Propositional:
- `cnf2smt`: CNF to SMT2 encoding
- `cnf2lp`: CNF to Linear Programming format
- `dimacs2smt`: DIMACS CNF to SMT2
- `wcnf2z3`: Weighted CNF to Z3 optimization

QBF (Quantified Boolean Formulas):
- `qbf2smt`: QBF to SMT2 encoding

SMT-LIB:
- `smt2c`: SMT-LIB to C code generation
- `smt2sympy`: SMT-LIB to SymPy expressions

SyGuS:
- `sygus2smt`: SyGuS syntax to SMT2

FlatZinc (from fzn2omt/):
- `fzn2z3`: FlatZinc to Z3
- `fzn2cvc4`: FlatZind to CVC4
- `fzn2optimathsat`: FlatZinc to Optimathsat
- `smt2model2fzn`: SMT model to FlatZinc solution

Example:
    >>> # Convert DIMACS CNF to SMT2
    >>> from aria.translator import dimacs2smt
    >>> # Parse and convert a DIMACS file to SMT2 format
    >>> dimacs2smt.convert_file('input.cnf', 'output.smt2')
    
    >>> # Convert QBF to SMT
    >>> from aria.translator import qbf2smt
    >>> qbf2smt.convert('input.qbf', 'output.smt2')
"""

from . import dimacs2smt
from . import cnf2smt
from . import cnf2lp
from . import qbf2smt
from . import smt2c
from . import smt2sympy
from . import sygus2smt
from . import wcnf2z3
from . import fzn2omt

__all__ = [
    "dimacs2smt",
    "cnf2smt", 
    "cnf2lp",
    "qbf2smt",
    "smt2c",
    "smt2sympy",
    "sygus2smt",
    "wcnf2z3",
    "fzn2omt",
]
