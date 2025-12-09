"""
MaxSMT solving algorithms module.

This module provides various algorithms for solving MaxSMT problems:
1. Core-guided approach (Fu-Malik / MSUS3) - SAT'13
2. Implicit Hitting Set approach (IHS) - IJCAR'18 
3. Local search for MaxSMT(LIA)
4. Z3's built-in Optimize engine

MaxSMT extends MaxSAT to SMT formulas, where each formula can have a weight 
(hard constraints have infinite weight, soft constraints have finite weights).
The goal is to find an assignment that maximizes the sum of weights of satisfied constraints.
"""

from typing import List, Tuple, Optional

import z3

from .base import MaxSMTAlgorithm, MaxSMTSolverBase
from .core_guided import CoreGuidedSolver
from .ihs import ImplicitHittingSetSolver
from .local_search import LocalSearchSolver
from .z3_optimize import Z3OptimizeSolver

# Export main classes and functions
__all__ = [
    'MaxSMTAlgorithm',
    'MaxSMTSolver',
    'solve_maxsmt',
    'demo',
    'example_scheduling',
    'example_minimal_correction'
]


class MaxSMTSolver:
    """
    Main class for solving MaxSMT problems
    """
    def __init__(self, algorithm: str = "core-guided", solver_name: str = "z3"):
        """Initialize the MaxSMT solver

        Args:
            algorithm: Algorithm to use (core-guided, ihs, local-search, z3-opt)
            solver_name: Name of the underlying SMT solver
        """
        self.algorithm = algorithm
        self.solver_name = solver_name
        self.hard_constraints = []
        self.soft_constraints = []
        self.weights = []
        
        # Create the appropriate solver based on the algorithm
        if algorithm == "z3-opt":
            self.solver = Z3OptimizeSolver(solver_name)
        elif algorithm == "core-guided" or algorithm == MaxSMTAlgorithm.CORE_GUIDED:
            self.solver = CoreGuidedSolver(solver_name)
        elif algorithm == "ihs" or algorithm == MaxSMTAlgorithm.IHS:
            self.solver = ImplicitHittingSetSolver(solver_name)
        elif algorithm == "local-search" or algorithm == MaxSMTAlgorithm.LOCAL_SEARCH:
            self.solver = LocalSearchSolver(solver_name)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def add_hard_constraint(self, constraint):
        """Add a hard constraint (must be satisfied)
        
        Args:
            constraint: SMT formula that must be satisfied
        """
        self.hard_constraints.append(constraint)
        self.solver.add_hard_constraint(constraint)
    
    def add_hard_constraints(self, constraints):
        """Add multiple hard constraints
        
        Args:
            constraints: List of SMT formulas that must be satisfied
        """
        self.hard_constraints.extend(constraints)
        self.solver.add_hard_constraints(constraints)

    def add_soft_constraint(self, constraint, weight: float = 1.0):
        """Add a soft constraint with a weight
        
        Args:
            constraint: SMT formula that should be satisfied if possible
            weight: Weight of the constraint (higher = more important)
        """
        self.soft_constraints.append(constraint)
        self.weights.append(weight)
        self.solver.add_soft_constraint(constraint, weight)
    
    def add_soft_constraints(self, constraints, weights=None):
        """Add multiple soft constraints with weights
        
        Args:
            constraints: List of SMT formulas 
            weights: List of weights (default: all 1.0)
        """
        if weights is None:
            weights = [1.0] * len(constraints)
        
        if len(constraints) != len(weights):
            raise ValueError("Number of constraints must match number of weights")
        
        self.soft_constraints.extend(constraints)
        self.weights.extend(weights)
        self.solver.add_soft_constraints(constraints, weights)
    
    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Solve the MaxSMT problem using the selected algorithm
        
        Returns:
            Tuple of (sat, model, optimal_cost)
            sat: True if a solution was found
            model: z3 model if sat is True, None otherwise
            optimal_cost: Sum of weights of unsatisfied soft constraints
        """
        return self.solver.solve()


def solve_maxsmt(hard_constraints: List[z3.ExprRef],
                 soft_constraints: List[z3.ExprRef],
                 weights: List[float] = None,
                 algorithm: str = "core-guided",
                 solver_name: str = "z3") -> Tuple[bool, Optional[z3.ModelRef], float]:
    """Convenience function to solve a MaxSMT problem
    
    Args:
        hard_constraints: List of hard constraints (must be satisfied)
        soft_constraints: List of soft constraints (may be violated)
        weights: List of weights for soft constraints (default: all 1.0)
        algorithm: Algorithm to use (core-guided, ihs, local-search, z3-opt)
        solver_name: Underlying SMT solver to use
    
    Returns:
        Tuple of (sat, model, optimal_cost)
        sat: True if a solution was found
        model: z3 model if sat is True, None otherwise
        optimal_cost: Sum of weights of unsatisfied soft constraints
    """
    if weights is None:
        weights = [1.0] * len(soft_constraints)
    
    solver = MaxSMTSolver(algorithm, solver_name)
    solver.add_hard_constraints(hard_constraints)
    solver.add_soft_constraints(soft_constraints, weights)
    
    return solver.solve()


def demo():
    """Demonstrate the MaxSMT solver with a simple example"""
    import time
    
    # Create variables
    x, y = z3.Ints('x y')
    
    # Hard constraints
    hard = [x >= 0, y >= 0, x + y <= 10]
    
    # Soft constraints with weights
    soft = [x == 5, y == 5, x + y == 8]
    weights = [1.0, 1.0, 2.0]
    
    # Solve using different algorithms
    algorithms = ["core-guided", "ihs", "local-search", "z3-opt"]
    
    for alg in algorithms:
        print(f"\nSolving with {alg}")
        start_time = time.time()
        sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm=alg)
        end_time = time.time()
        
        if sat:
            print(f"Model: x = {model.eval(x)}, y = {model.eval(y)}")
            print(f"Cost: {cost}")
        else:
            print("Unsatisfiable")
        
        print(f"Time: {end_time - start_time:.4f} seconds")


def example_scheduling():
    """Example: Job scheduling problem with MaxSMT
    
    We have jobs with durations and deadlines.
    Hard constraints: Jobs cannot overlap
    Soft constraints: Try to meet deadlines
    """
    print("\n=== Job Scheduling Example ===")
    
    # Number of jobs
    n_jobs = 5
    
    # Job durations (time units)
    durations = [3, 2, 4, 1, 5]
    
    # Job deadlines
    deadlines = [5, 4, 10, 3, 12]
    
    # Create variables for start times
    starts = [z3.Int(f"start_{i}") for i in range(n_jobs)]
    
    # Hard constraints
    hard = []
    
    # All jobs must start at or after time 0
    for i in range(n_jobs):
        hard.append(starts[i] >= 0)
    
    # Jobs cannot overlap (for each pair of jobs)
    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
            # Either job i finishes before job j starts, or vice versa
            hard.append(z3.Or(
                starts[i] + durations[i] <= starts[j],  # job i before job j
                starts[j] + durations[j] <= starts[i]   # job j before job i
            ))
    
    # Soft constraints: try to meet deadlines
    soft = []
    weights = []
    
    for i in range(n_jobs):
        # Try to finish before deadline
        soft.append(starts[i] + durations[i] <= deadlines[i])
        # Assign weights based on job importance (for this example, all equal)
        weights.append(1.0)
    
    # Solve using different algorithms
    algorithms = ["core-guided", "z3-opt"]
    all_results = {}
    
    for alg in algorithms:
        print(f"\nSolving scheduling problem with {alg}")
        start_time = time.time()
        sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm=alg)
        end_time = time.time()
        
        if sat:
            print("Optimal Schedule:")
            jobs_info = [(i, model.eval(starts[i]).as_long(), 
                         model.eval(starts[i]).as_long() + durations[i],
                         deadlines[i]) for i in range(n_jobs)]
            
            # Store results for comparison
            all_results[alg] = {
                'cost': cost,
                'jobs_info': jobs_info
            }
            
            # Sort by start time
            jobs_info.sort(key=lambda x: x[1])
            
            print("Job  Start  End  Deadline  Status")
            print("----  -----  ---  --------  ------")
            missed = 0
            for job, start, end, deadline in jobs_info:
                status = "On time" if end <= deadline else "Late"
                if status == "Late":
                    missed += 1
                print(f"{job:3}   {start:4}  {end:3}   {deadline:6}    {status}")
                
            print(f"\nMissed deadlines: {missed}")
            print(f"Cost: {cost}")
        else:
            print("Unsatisfiable")
        
        print(f"Time: {end_time - start_time:.4f} seconds")
    
    # Compare results if we have both algorithms
    if len(all_results) == 2:
        print("\n=== Comparing Results Between Algorithms ===")
        
        # Extract results for each algorithm
        cg_results = all_results.get("core-guided")
        z3_results = all_results.get("z3-opt")
        
        if cg_results and z3_results:
            print(f"Core-guided cost: {cg_results['cost']}, Z3-Opt cost: {z3_results['cost']}")
            
            if abs(cg_results['cost'] - z3_results['cost']) < 1e-6:
                print("Both algorithms found solutions with the same cost.")
            else:
                print("The algorithms found solutions with different costs!")
                
                # Which constraints are violated in each solution?
                print("\nSoft constraints violated in each solution:")
                print("Constraint | Core-guided | Z3-Opt")
                print("-----------+-------------+-------")
                
                # Get job results by job ID for easier comparison
                cg_jobs = {job[0]: job for job in cg_results['jobs_info']}
                z3_jobs = {job[0]: job for job in z3_results['jobs_info']}
                
                for i in range(n_jobs):
                    cg_job = cg_jobs[i]
                    z3_job = z3_jobs[i]
                    
                    cg_end = cg_job[2]
                    z3_end = z3_job[2]
                    deadline = deadlines[i]
                    
                    cg_violation = "Late" if cg_end > deadline else "On time"
                    z3_violation = "Late" if z3_end > deadline else "On time"
                    
                    print(f"Job {i:3}    | {cg_violation:11} | {z3_violation}")
                
                print("\nDetailed job comparison:")
                print("Job | Core-guided (Start-End) | Z3-Opt (Start-End) | Deadline")
                print("----+------------------------+------------------+--------")
                
                for i in range(n_jobs):
                    cg_job = cg_jobs[i]
                    z3_job = z3_jobs[i]
                    
                    cg_start = cg_job[1]
                    cg_end = cg_job[2]
                    z3_start = z3_job[1]
                    z3_end = z3_job[2]
                    deadline = deadlines[i]
                    
                    print(f"{i:3} | {cg_start:2}-{cg_end:2} {' (Late)' if cg_end > deadline else '       '} | "
                          f"{z3_start:2}-{z3_end:2} {' (Late)' if z3_end > deadline else '       '} | {deadline}")


def example_minimal_correction():
    """Example: Minimal correction subset
    
    Find a minimal subset of clauses to remove to make an unsatisfiable formula satisfiable.
    This is the dual of the MaxSAT problem and is useful in debugging.
    """
    print("\n=== Minimal Correction Subset Example ===")
    
    # Create variables
    a, b, c, d = z3.Bools('a b c d')
    
    # Create a set of clauses that is unsatisfiable
    # (a AND b AND NOT a) is unsatisfiable
    clauses = [
        a,             # clause 1
        b,             # clause 2
        z3.Not(a),     # clause 3
        z3.Implies(b, c),  # clause 4
        z3.Implies(c, d)   # clause 5
    ]
    
    # No hard constraints
    hard = []
    
    # Each clause is a soft constraint
    soft = clauses
    
    # All clauses have equal weight
    weights = [1.0] * len(clauses)
    
    # Solve using Z3's optimize engine
    print("\nFinding minimal correction subset")
    sat, model, cost = solve_maxsmt(hard, soft, weights, algorithm="z3-opt")
    
    if sat:
        print("\nSolution found by removing these clauses:")
        
        for i, clause in enumerate(clauses):
            if not model.evaluate(clause):
                print(f"Clause {i+1}: {clause}")
        
        print(f"\nCost (number of clauses removed): {cost}")
        print(f"Model: a={model.eval(a)}, b={model.eval(b)}, c={model.eval(c)}, d={model.eval(d)}")
    else:
        print("Unexpected: Could not find a solution")


if __name__ == '__main__':
    print("\n=== Basic Example ===")
    demo()
    example_scheduling()
    example_minimal_correction() 