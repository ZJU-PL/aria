# coding: utf-8
"""
Algorithms for computing backbones of SAT formulas.

This module provides several algorithms for computing backbones of SAT formulas:

1. Iterative Algorithm: The most straightforward approach that tests each
   variable one by one using a SAT solver.
2. Chunking Algorithm: Improves efficiency by testing multiple variables
   at once in "chunks".
3. Backbone Refinement Algorithm: Uses a refinement-based approach that
   leverages previous models to quickly identify backbone literals.
"""

import logging
from enum import Enum
from typing import List, Tuple

from pysat.formula import CNF

from aria.bool.sat.pysat_solver import PySATSolver
from aria.utils.types import SolverResult

logger = logging.getLogger(__name__)


class BackboneAlgorithm(Enum):
    """Enumeration of backbone computation algorithms."""

    ITERATIVE = "iterative"
    CHUNKING = "chunking"
    BACKBONE_REFINEMENT = "backbone_refinement"


def compute_backbone(
    cnf: CNF,
    algorithm: BackboneAlgorithm = BackboneAlgorithm.BACKBONE_REFINEMENT,
    solver_name: str = "cd",
    chunk_size: int = 10,
) -> Tuple[List[int], int]:
    """Compute the backbone of a SAT formula."""
    if algorithm == BackboneAlgorithm.ITERATIVE:
        return compute_backbone_iterative(cnf, solver_name)
    if algorithm == BackboneAlgorithm.CHUNKING:
        return compute_backbone_chunking(cnf, solver_name, chunk_size)
    if algorithm == BackboneAlgorithm.BACKBONE_REFINEMENT:
        return compute_backbone_refinement(cnf, solver_name)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_backbone_iterative(
    cnf: CNF, solver_name: str = "cd"
) -> Tuple[List[int], int]:
    """Compute the backbone of a SAT formula using iterative literal checks."""
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        return [], 1

    backbone_literals = []
    num_solver_calls = 1
    variables = set(abs(lit) for clause in cnf.clauses for lit in clause)

    for var in variables:
        result = solver.check_sat_assuming([-var])
        num_solver_calls += 1
        if result == SolverResult.UNSAT:
            backbone_literals.append(var)
            continue

        result = solver.check_sat_assuming([var])
        num_solver_calls += 1
        if result == SolverResult.UNSAT:
            backbone_literals.append(-var)

    return backbone_literals, num_solver_calls


def compute_backbone_chunking(
    cnf: CNF, solver_name: str = "cd", chunk_size: int = 10
) -> Tuple[List[int], int]:
    """Compute the backbone using chunked assumption flips."""
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        return [], 1

    model = solver.get_model()
    model_dict = {abs(lit): lit > 0 for lit in model}
    backbone_literals = []
    num_solver_calls = 1
    variables = list(set(abs(lit) for clause in cnf.clauses for lit in clause))

    for i in range(0, len(variables), chunk_size):
        chunk = variables[i : i + chunk_size]
        assumptions = []
        for var in chunk:
            if var in model_dict:
                assumptions.append(-var if model_dict[var] else var)
            else:
                assumptions.append(var)

        result = solver.check_sat_assuming(assumptions)
        num_solver_calls += 1
        if result != SolverResult.UNSAT:
            continue

        for var in chunk:
            if var not in model_dict:
                continue
            assumption = -var if model_dict[var] else var
            result = solver.check_sat_assuming([assumption])
            num_solver_calls += 1
            if result == SolverResult.UNSAT:
                backbone_literals.append(var if model_dict[var] else -var)

    return backbone_literals, num_solver_calls


def compute_backbone_refinement(
    cnf: CNF, solver_name: str = "cd"
) -> Tuple[List[int], int]:
    """Compute the backbone using model refinement."""
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        return [], 1

    model = solver.get_model()
    potential_backbone = set(model)
    backbone_literals = []
    num_solver_calls = 1

    while potential_backbone:
        lit = next(iter(potential_backbone))
        potential_backbone.remove(lit)
        result = solver.check_sat_assuming([-lit])
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            backbone_literals.append(lit)
            continue

        new_model = solver.get_model()
        potential_backbone &= set(new_model)

    return backbone_literals, num_solver_calls


def compute_backbone_with_approximation(
    cnf: CNF, solver_name: str = "cd"
) -> Tuple[List[int], List[int], int]:
    """Compute definite and potential backbone literals."""
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        return [], [], 1

    num_models = min(10, 2 ** min(10, cnf.nv))
    previous_reduce_samples = solver.reduce_samples
    solver.reduce_samples = False
    try:
        models = solver.sample_models(num_models)
    finally:
        solver.reduce_samples = previous_reduce_samples

    num_solver_calls = 1 + len(models)
    if not models:
        return [], [], num_solver_calls

    common_literals = set(models[0])
    for model in models[1:]:
        common_literals &= set(model)

    potential_backbone = sorted(common_literals, key=lambda lit: (abs(lit), lit < 0))
    definite_backbone = []
    for lit in potential_backbone:
        result = solver.check_sat_assuming([-lit])
        num_solver_calls += 1
        if result == SolverResult.UNSAT:
            definite_backbone.append(lit)

    return definite_backbone, potential_backbone, num_solver_calls


def is_backbone_literal(
    cnf: CNF, literal: int, solver_name: str = "cd"
) -> Tuple[bool, int]:
    """Check whether a literal belongs to the backbone."""
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        return False, 1

    result = solver.check_sat_assuming([-literal])
    return result == SolverResult.UNSAT, 2
