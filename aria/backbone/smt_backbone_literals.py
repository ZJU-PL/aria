"""
SMT Backbone literals generation via lazy model enumeration

A backbone literal is a literal that is entailed by the formula.
For example, in the formula "p1 or p2 or p3", "p1" is a backbone literal.
"""

from typing import List

from z3 import Solver, Not, And, unsat, sat, ExprRef


# from aria.utils.z3_expr_utils import get_atoms


def get_backbone_literals_by_sequence_checking(fml: ExprRef, literals: List[ExprRef]):
    """
    This function takes an expression and a list of atoms.
    It returns a list of all the literals that are entailed by the expression,
    including both positive and negative literals.

    :param fml: The SMT formula F
    :param literals: The set of literals to be considered {l1,...,ln}
    :return: A list of literals that are entailed by F

    NOTE: If li is a backbone (semantic consequence) of F,
    then li evaluates to true under every model of F.
    """
    res = []
    solver = Solver()
    solver.add(fml)

    for literal in literals:
        # Check if the positive literal is a backbone
        solver.push()
        solver.add(Not(literal))
        if solver.check() == unsat:
            res.append(literal)
        solver.pop()


def get_backbone_literals_by_model_enumeration(fml: ExprRef, literals: List[ExprRef]):
    """
    A backbone literal is a literal that is entailed by the formula.

    The idea: enumerate all models of the fml and check whether each literal is true in each model.
    If not, we can prune the literal in that round.
    :param fml:
    :param literals: the set of literals to be considered
    :return: the set of literal that is entailed by fml
    """
    res = set(literals)
    solver = Solver()
    solver.add(fml)
    while solver.check() == sat:
        model = solver.model()
        for literal in literals:
            if literal not in res:
                if not model.eval(literal, model_completion=True):
                    # Bug fix: Remove the literal to be pruned from the result set
                    # remove the literal from res
                    res.remove(literal)
        solver.add(Not(And([literal == model[literal] for literal in literals])))

    return res


def get_backbone_literals_by_unsat_core_enumeration(
    fml: ExprRef, literals: List[ExprRef]
):
    """
    A backbone literal is a literal that is entailed by the formula.
    The idea: enumerate all models of the fml and check whether each literal is true in each model.
    """
    raise NotImplementedError


def get_backbone_literals_by_monadic_predicate_abstraction(
    fml: ExprRef, literals: List[ExprRef]
):
    """Call the monadic predicate abstraction algorithm to get the backbone literals."""
    raise NotImplementedError


def get_backbone_literals(fml: ExprRef, literals: List[ExprRef], alg: str):
    """
    This function takes an expression and a list of atoms.
    It returns a list of all the literals that are entailed by the expression,
    including both positive and negative literals.

    :param fml: The SMT formula F
    :param literals: The set of literals to be considered {l1,...,ln}
    :param alg: The algorithm to use
    """
    # allow for choosing different implementations in this file
    if alg == "sequence_checking":
        return get_backbone_literals_by_sequence_checking(fml, literals)
    if alg == "model_enumeration":
        return get_backbone_literals_by_model_enumeration(fml, literals)
    if alg == "unsat_core_enumeration":
        return get_backbone_literals_by_unsat_core_enumeration(fml, literals)
    raise NotImplementedError
