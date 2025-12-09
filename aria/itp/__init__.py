"""
This module provides a interactive proof assistant for the Z3 theorem prover (NOTE: not "itp" does not mean "interpolant":)).

It includes the following features:
- Proofs
- Axioms
- Definitions

(Note: here, "ITP" means "Interactive Theorem Proving", not "Interpolation")
"""

from . import smt
from . import kernel
from . import notation
from . import utils
from . import datatype
from . import rewrite
from . import tactics

Proof = kernel.Proof

prove = tactics.prove

axiom = kernel.axiom

define = kernel.define

FreshVar = kernel.FreshVar

def FreshVars(names: str, sort: smt.SortRef) -> tuple:
    """
    Create multiple fresh variables from a space-separated string of names.

    Args:
        names: Space-separated string of variable names
        sort: The sort for all variables

    Returns:
        tuple: Tuple of fresh variables

    >>> x, y, z = FreshVars("x y z", smt.IntSort())
    """
    return tuple(FreshVar(name.strip(), sort) for name in names.split() if name.strip())

QForAll = notation.QForAll

QExists = notation.QExists

cond = notation.cond

Inductive = kernel.Inductive

Struct = datatype.Struct

NewType = datatype.NewType

InductiveRel = datatype.InductiveRel

Enum = datatype.Enum

Calc = tactics.Calc

Lemma = tactics.Lemma

Theorem = tactics.Theorem
PTheorem = tactics.PTheorem

simp = rewrite.simp

search = utils.search

# TODO: Remove this
R = smt.RealSort()
Z = smt.IntSort()
RSeq = Z >> R
RFun = R >> R

__all__ = [
    "prove",
    "axiom",
    "define",
    "Proof",
    "FreshVar",
    "FreshVars",
    "QForAll",
    "QExists",
    "cond",
    "Struct",
    "NewType",
    "Inductive",
    "Calc",
    "Lemma",
    "Theorem",
    "PTheorem",
    "R",
    "Z",
    "RSeq",
    "RFun",
    "simp",
    "search",
]
