"""
Solving SAT problems via brute-force enumeration
"""
from typing import List, Dict, Set, Optional, Union
import time
import multiprocessing as mp
from math import ceil

Formula = List[List[int]]
Assignment = Dict[int, bool]


def solve_sat_brute_force(formula: Formula, variables: Union[List[int], Set[int]]) -> Optional[Assignment]:
    """
    Solve SAT problem using brute force enumeration with early termination
    
    Args:
        formula: Boolean formula in CNF form (list of clauses)
        variables: List or set of variables in the formula
    
    Returns:
        dict: Solution mapping variables to boolean values if satisfiable
        None: If formula is unsatisfiable
    """
    # Convert variables to sorted list for consistent iteration
    variables = sorted(set(variables))
    num_vars = len(variables)
    var_to_index = {var: idx for idx, var in enumerate(variables)}

    # Pre-process formula for faster access
    pos_occurrences: Dict[int, List[int]] = {var: [] for var in variables}
    neg_occurrences: Dict[int, List[int]] = {var: [] for var in variables}

    for clause_idx, clause in enumerate(formula):
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                pos_occurrences[var].append(clause_idx)
            else:
                neg_occurrences[var].append(clause_idx)

    # Try all possible assignments
    for i in range(2 ** num_vars):
        # Generate assignment from binary representation
        assignment = {}
        clause_status = [False] * len(formula)

        # Process assignment and evaluate clauses simultaneously
        for j in range(num_vars):
            var = variables[j]
            val = bool((i >> j) & 1)
            assignment[var] = val

            # Update affected clauses
            if val:
                for clause_idx in pos_occurrences[var]:
                    clause_status[clause_idx] = True
            else:
                for clause_idx in neg_occurrences[var]:
                    clause_status[clause_idx] = True

        # Check if all clauses are satisfied
        if all(clause_status):
            return assignment

    return None


def check_range(start: int, end: int, formula: Formula, variables: List[int],
                pos_occurrences: Dict[int, List[int]],
                neg_occurrences: Dict[int, List[int]]) -> Optional[Assignment]:
    """Helper function to check a range of assignments"""
    num_vars = len(variables)
    for i in range(start, end):
        assignment = {}
        clause_status = [False] * len(formula)

        for j in range(num_vars):
            var = variables[j]
            val = bool((i >> j) & 1)
            assignment[var] = val

            if val:
                for clause_idx in pos_occurrences[var]:
                    clause_status[clause_idx] = True
            else:
                for clause_idx in neg_occurrences[var]:
                    clause_status[clause_idx] = True

        if all(clause_status):
            return assignment
    return None


def solve_sat_brute_force_parallel(formula: Formula, variables: Union[List[int], Set[int]],
                                   num_processes: Optional[int] = None) -> Optional[Assignment]:
    """
    Parallel version of SAT solver using brute force enumeration
    """
    variables = sorted(set(variables))
    num_vars = len(variables)
    total_assignments = 2 ** num_vars

    if num_processes is None:
        num_processes = mp.cpu_count()

    # Pre-process formula
    pos_occurrences: Dict[int, List[int]] = {var: [] for var in variables}
    neg_occurrences: Dict[int, List[int]] = {var: [] for var in variables}

    for clause_idx, clause in enumerate(formula):
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                pos_occurrences[var].append(clause_idx)
            else:
                neg_occurrences[var].append(clause_idx)

    # Split work into chunks
    chunk_size = ceil(total_assignments / num_processes)
    ranges = [(i * chunk_size, min((i + 1) * chunk_size, total_assignments),
               formula, variables, pos_occurrences, neg_occurrences)
              for i in range(num_processes)]

    # Create pool and run parallel search
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(check_range, ranges)

    # Return first satisfying assignment found
    for result in results:
        if result is not None:
            return result

    return None


def evaluate(formula: Formula, assignment: Assignment) -> bool:
    """
    Evaluate a boolean formula under a given assignment
    
    Args:
        formula: Boolean formula in CNF form
        assignment: Dictionary mapping variables to boolean values
        
    Returns:
        bool: True if formula is satisfied, False otherwise
    """
    return all(
        any(
            assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]
            for lit in clause
        )
        for clause in formula
    )


def test() -> None:
    """Test both sequential and parallel SAT solvers"""
    test_cases = [
        # Simple satisfiable case
        ([[1, 2], [-1, 2]], {1, 2}),
        # Simple unsatisfiable case
        ([[1], [-1]], {1}),
        # More complex satisfiable case
        ([[1, 2, 3], [-1, -2], [2, -3]], {1, 2, 3}),
        # Add a larger test case to better demonstrate parallel speedup
        ([[1, 2, 3, 4], [-1, -2], [2, -3], [3, -4], [-2, 4]], {1, 2, 3, 4}),
    ]

    for formula, variables in test_cases:
        print(f"\nTesting formula: {formula}")

        # Test sequential version
        start_time = time.time()
        result_seq = solve_sat_brute_force(formula, variables)
        seq_time = time.time() - start_time
        print(f"Sequential result: {result_seq}")
        print(f"Sequential time: {seq_time:.4f} seconds")

        # Test parallel version
        start_time = time.time()
        result_par = solve_sat_brute_force_parallel(formula, variables)
        par_time = time.time() - start_time
        print(f"Parallel result: {result_par}")
        print(f"Parallel time: {par_time:.4f} seconds")
        print(f"Speedup: {seq_time / par_time:.2f}x")

        if result_par is not None:
            print(f"Verification: {evaluate(formula, result_par)}")
        print(f"Time taken: {par_time:.4f} seconds")  # Fixed: using par_time instead of undefined end_time


if __name__ == "__main__":
    test()
