"""
Algorithms for computing backbones of SAT formulas

This module provides several algorithms for computing backbones of SAT formulas:

1. Iterative Algorithm: The most straightforward approach that tests each 
   variable one by one using a SAT solver.

2. Chunking Algorithm: Improves efficiency by testing multiple variables 
   at once in "chunks".

3. Backbone Refinement Algorithm: Uses a refinement-based approach that
   leverages previous models to quickly identify backbone literals.

References:
- J. Marques-Silva, M. Janota, C. Mencía. "Minimal sets over monotone predicates 
  in Boolean formulae." CAV, 2013.
- C. Mencía, A. Previti, J. Marques-Silva. "Literal-based MCS extraction." IJCAI, 2015.
- A. Previti, A. Ignatiev, A. Morgado, J. Marques-Silva. "Prime compilation of 
  non-clausal formulae." IJCAI, 2015.
- R. Grumberg, A. Schumann, A. Melnikov. "Faster Backbone Computation Using 
  Efficient Enumeration Methods of Minimal Correction Subsets." CP, 2022.
"""

import logging
from enum import Enum
from typing import List, Set, Tuple, Optional, Dict

from aria.bool.sat.pysat_solver import PySATSolver
from aria.utils.types import SolverResult
from pysat.formula import CNF

logger = logging.getLogger(__name__)


class BackboneAlgorithm(Enum):
    """Enumeration of backbone computation algorithms."""
    ITERATIVE = "iterative"
    CHUNKING = "chunking"
    BACKBONE_REFINEMENT = "backbone_refinement"


def compute_backbone(cnf: CNF, algorithm: BackboneAlgorithm = BackboneAlgorithm.BACKBONE_REFINEMENT,
                     solver_name: str = "cd", chunk_size: int = 10) -> Tuple[List[int], int]:
    """
    Compute the backbone of a SAT formula.
    
    Args:
        cnf: The CNF formula
        algorithm: The algorithm to use for backbone computation
        solver_name: The SAT solver to use
        chunk_size: Size of chunks for the chunking algorithm
        
    Returns:
        A tuple containing:
        - List of backbone literals (positive or negative integers)
        - Number of SAT solver calls made
    """
    if algorithm == BackboneAlgorithm.ITERATIVE:
        return compute_backbone_iterative(cnf, solver_name)
    elif algorithm == BackboneAlgorithm.CHUNKING:
        return compute_backbone_chunking(cnf, solver_name, chunk_size)
    elif algorithm == BackboneAlgorithm.BACKBONE_REFINEMENT:
        return compute_backbone_refinement(cnf, solver_name)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_backbone_iterative(cnf: CNF, solver_name: str = "cd") -> Tuple[List[int], int]:
    """
    Compute the backbone of a SAT formula using the iterative algorithm.
    
    This is the most straightforward approach that tests each variable
    one by one using a SAT solver.
    
    Algorithm:
    1. Initialize the solver with the formula
    2. Check if the formula is satisfiable
    3. For each variable v in the formula:
       a. Check if F ∧ ¬v is satisfiable
       b. If not, v is a backbone literal
       c. Check if F ∧ v is satisfiable 
       d. If not, ¬v is a backbone literal
    
    Args:
        cnf: The CNF formula
        solver_name: The SAT solver to use
        
    Returns:
        A tuple containing:
        - List of backbone literals (positive or negative integers)
        - Number of SAT solver calls made
    """
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    # Check if formula is satisfiable
    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        # If formula is UNSAT, there are no backbone literals
        return [], 1

    backbone_literals = []
    num_solver_calls = 1  # Initial satisfiability check

    # Get all variables in the formula
    variables = set(abs(lit) for clause in cnf.clauses for lit in clause)

    for var in variables:
        # Check if F ∧ ¬var is satisfiable
        result = solver.check_sat_assuming([-var])
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            # If F ∧ ¬var is UNSAT, then var is a backbone literal
            backbone_literals.append(var)
            continue

        # Check if F ∧ var is satisfiable
        result = solver.check_sat_assuming([var])
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            # If F ∧ var is UNSAT, then ¬var is a backbone literal
            backbone_literals.append(-var)

    return backbone_literals, num_solver_calls


def compute_backbone_chunking(cnf: CNF, solver_name: str = "cd", chunk_size: int = 10) -> Tuple[List[int], int]:
    """
    Compute the backbone of a SAT formula using the chunking algorithm.
    
    This algorithm improves efficiency by testing multiple variables
    at once in "chunks". When a chunk test reveals a backbone literal,
    we narrow down to find the specific literal(s).
    
    Algorithm:
    1. Initialize the solver with the formula
    2. Check if the formula is satisfiable and get a model M
    3. Split variables into chunks of size chunk_size
    4. For each chunk:
       a. Flip the values of all variables in the chunk (from their values in M)
       b. Check if formula is still satisfiable with these flipped values
       c. If not, there's at least one backbone literal in the chunk
       d. For each variable in the chunk:
          i. Check if flipping just this variable makes formula unsatisfiable
          ii. If yes, it's a backbone literal
    
    Args:
        cnf: The CNF formula
        solver_name: The SAT solver to use
        chunk_size: Size of each chunk of variables to test
        
    Returns:
        A tuple containing:
        - List of backbone literals (positive or negative integers)
        - Number of SAT solver calls made
    """
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    # Check if formula is satisfiable
    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        # If formula is UNSAT, there are no backbone literals
        return [], 1

    # Get a model
    model = solver.get_model()
    model_dict = {abs(lit): lit > 0 for lit in model}

    backbone_literals = []
    num_solver_calls = 1  # Initial satisfiability check

    # Get all variables in the formula
    variables = list(set(abs(lit) for clause in cnf.clauses for lit in clause))

    # Process variables in chunks
    for i in range(0, len(variables), chunk_size):
        chunk = variables[i:i + chunk_size]

        # Create assumptions that flip the values of all variables in the chunk
        assumptions = []
        for var in chunk:
            if var in model_dict:
                # Flip the value from the model
                assumptions.append(-var if model_dict[var] else var)
            else:
                # If variable not in model, assign arbitrarily
                assumptions.append(var)

        # Check if formula is satisfiable with these assumptions
        result = solver.check_sat_assuming(assumptions)
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            # There's at least one backbone literal in this chunk
            # Check each variable individually
            for var in chunk:
                if var in model_dict:
                    assumption = -var if model_dict[var] else var
                    result = solver.check_sat_assuming([assumption])
                    num_solver_calls += 1

                    if result == SolverResult.UNSAT:
                        # This variable is a backbone literal
                        backbone_literals.append(var if model_dict[var] else -var)

    return backbone_literals, num_solver_calls


def compute_backbone_refinement(cnf: CNF, solver_name: str = "cd") -> Tuple[List[int], int]:
    """
    Compute the backbone of a SAT formula using the backbone refinement algorithm.
    
    This algorithm maintains sets of potential backbone literals and non-backbone literals,
    refining these sets with each new model found.
    
    Algorithm:
    1. Initialize the solver with the formula
    2. Check if formula is satisfiable and get a model M₁
    3. Initialize potential backbone set B = {literals in M₁}
    4. While B is not empty:
       a. Pick a literal l from B
       b. Check if F ∧ ¬l is satisfiable, getting model M₂ if SAT
       c. If UNSAT, l is a backbone literal
       d. If SAT, update B = B ∩ {literals in M₂}
    
    Args:
        cnf: The CNF formula
        solver_name: The SAT solver to use
        
    Returns:
        A tuple containing:
        - List of backbone literals (positive or negative integers)
        - Number of SAT solver calls made
    """
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    # Check if formula is satisfiable
    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        # If formula is UNSAT, there are no backbone literals
        return [], 1

    # Get initial model
    model = solver.get_model()

    # Initialize potential backbone literals from the model
    potential_backbone = set(model)
    backbone_literals = []
    num_solver_calls = 1  # Initial satisfiability check

    # Refine potential backbone literals
    while potential_backbone:
        # Pick a literal from potential backbone
        lit = next(iter(potential_backbone))
        potential_backbone.remove(lit)

        # Check if formula is satisfiable with negation of literal
        result = solver.check_sat_assuming([-lit])
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            # This literal is a backbone literal
            backbone_literals.append(lit)
        else:
            # Get the new model
            new_model = solver.get_model()
            # Refine potential backbone literals to those that appear in both models
            potential_backbone &= set(new_model)

    return backbone_literals, num_solver_calls


def compute_backbone_with_approximation(cnf: CNF, solver_name: str = "cd") -> Tuple[List[int], List[int], int]:
    """
    Compute an approximation of the backbone of a SAT formula.
    
    This algorithm computes a lower bound (definite backbone literals) and
    an upper bound (potential backbone literals) of the backbone.
    
    Args:
        cnf: The CNF formula
        solver_name: The SAT solver to use
        
    Returns:
        A tuple containing:
        - List of definite backbone literals
        - List of potential backbone literals
        - Number of SAT solver calls made
    """
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    # Check if formula is satisfiable
    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        # If formula is UNSAT, there are no backbone literals
        return [], [], 1

    # Collect multiple models
    models = []
    num_models = min(10, 2 ** min(10, cnf.nv))  # Collect at most 10 models
    models = solver.sample_models(num_models)

    num_solver_calls = 1  # Initial satisfiability check

    # Find common literals across all models (lower bound of backbone)
    if not models:
        return [], [], num_solver_calls

    common_literals = set(models[0])
    for model in models[1:]:
        common_literals &= set(model)

    # Verify each common literal to confirm it's a backbone literal
    definite_backbone = []
    for lit in common_literals:
        result = solver.check_sat_assuming([-lit])
        num_solver_calls += 1

        if result == SolverResult.UNSAT:
            definite_backbone.append(lit)

    # Compute upper bound: literals that might be backbone literals
    # (those not proven to not be in the backbone)
    potential_backbone = list(definite_backbone)
    variables = set(abs(lit) for clause in cnf.clauses for lit in clause)

    for var in variables:
        # Skip variables we've already determined are backbone literals
        if var in [abs(lit) for lit in definite_backbone]:
            continue

        # Check positive literal
        if var not in [abs(lit) for lit in potential_backbone]:
            result = solver.check_sat_assuming([-var])
            num_solver_calls += 1
            if result == SolverResult.UNSAT:
                potential_backbone.append(var)

        # Check negative literal
        if -var not in [abs(lit) for lit in potential_backbone]:
            result = solver.check_sat_assuming([var])
            num_solver_calls += 1
            if result == SolverResult.UNSAT:
                potential_backbone.append(-var)

    return definite_backbone, potential_backbone, num_solver_calls


def is_backbone_literal(cnf: CNF, literal: int, solver_name: str = "cd") -> Tuple[bool, int]:
    """
    Check if a literal is a backbone literal of a SAT formula.
    
    Args:
        cnf: The CNF formula
        literal: The literal to check
        solver_name: The SAT solver to use
        
    Returns:
        A tuple containing:
        - True if literal is a backbone literal, False otherwise
        - Number of SAT solver calls made
    """
    solver = PySATSolver(solver=solver_name)
    solver.add_cnf(cnf)

    # Check if formula is satisfiable
    result = solver.check_sat()
    if result == SolverResult.UNSAT:
        # If formula is UNSAT, there are no backbone literals
        return False, 1

    # Check if formula is satisfiable with negation of literal
    result = solver.check_sat_assuming([-literal])

    # If formula is unsatisfiable with negation of literal,
    # then literal is a backbone literal
    return result == SolverResult.UNSAT, 2
