#!/usr/bin/env python3
"""
Simple performance and solution quality comparison of different MaxSMT algorithms.

This file compares the four algorithms implemented in the MaxSMT module:
1. Core-guided approach (Fu-Malik / MSUS3)
2. Implicit Hitting Set approach (IHS)
3. Local search for MaxSMT(LIA)
4. Z3's built-in Optimize engine
"""

import time
import sys
import z3

from aria.optimization.maxsmt import MaxSMTSolver, solve_maxsmt


def calculate_cost(model, soft_constraints, weights):
    """Calculate the cost (sum of weights of violated constraints)"""
    total_violated_weight = 0.0
    for constraint, weight in zip(soft_constraints, weights):
        if not model.evaluate(constraint):
            total_violated_weight += weight
    return total_violated_weight


def test_algorithm(hard, soft, weights, algorithm, timeout=30):
    """Test a specific algorithm on a problem instance"""
    print(f"Running {algorithm}...", end="", flush=True)
    start_time = time.time()
    
    try:
        sat, model, reported_cost = solve_maxsmt(
            hard, soft, weights, algorithm=algorithm, solver_name="z3"
        )
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f" timed out after {elapsed:.2f}s")
            return False, None, float('inf'), elapsed
        
        if sat:
            standard_cost = calculate_cost(model, soft, weights)            
            print(f" done in {elapsed:.2f}s, cost={standard_cost:.2f}")
            return True, model, standard_cost, elapsed
        else:
            print(f" unsatisfiable, time={elapsed:.2f}s")
            return False, None, float('inf'), elapsed
            
    except Exception as e:
        print(f" failed: {str(e)}")
        return False, None, float('inf'), timeout


def generate_scheduling_problem():
    """Generate a scheduling problem for testing"""
    # Create 5 jobs with specific durations
    n_jobs = 5
    durations = [3, 2, 4, 1, 5]
    deadlines = [5, 4, 8, 3, 10]
    
    # Create variables for start times
    starts = [z3.Int(f"start_{i}") for i in range(n_jobs)]
    
    # Hard constraints
    hard = []
    
    # All jobs must start at or after time 0
    for i in range(n_jobs):
        hard.append(starts[i] >= 0)
    
    # Jobs cannot overlap
    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
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
        weights.append(1.0)
    
    return hard, soft, weights, starts


def generate_boolean_problem():
    """Generate a boolean problem for testing"""
    a, b, c, d, e = z3.Bools('a b c d e')
    
    # Hard constraints
    hard = [
        z3.Implies(a, b),
        z3.Implies(b, c),
        z3.Implies(c, z3.Or(d, e)),
        z3.Implies(d, z3.Not(e)),
    ]
    
    # Soft constraints - intentionally conflicting
    soft = [
        a,           # prefer a to be true
        z3.Not(b),   # prefer b to be false
        c,           # prefer c to be true
        d,           # prefer d to be true
        e,           # prefer e to be true
    ]
    
    weights = [3.0, 1.0, 2.0, 1.5, 1.5]
    
    return hard, soft, weights, (a, b, c, d, e)


def generate_integer_problem():
    """Generate an integer linear arithmetic problem for testing"""
    x, y, z = z3.Ints('x y z')
    
    # Hard constraints
    hard = [
        x >= 0, x <= 10,
        y >= 0, y <= 10,
        z >= 0, z <= 10,
        x + y + z <= 20,
        x <= y + 5,
    ]
    
    # Soft constraints - some conflicting targets
    soft = [
        x == 8,
        y == 7,
        z == 9,
        x + y == 10,
        y + z == 12,
        x == y,
        z == x + 2,
    ]
    
    weights = [3.0, 2.0, 2.0, 1.0, 1.0, 0.5, 0.5]
    
    return hard, soft, weights, (x, y, z)


def compare_models(algorithm_names, models, problem_vars):
    """Compare models from different algorithms"""
    print("\nVariable assignments by algorithm:")
    
    # For each variable, show values assigned by each algorithm
    if isinstance(problem_vars, tuple):
        # For explicitly defined variables (boolean, integer)
        for i, var in enumerate(problem_vars):
            values = []
            for alg, model in zip(algorithm_names, models):
                if model:
                    values.append(f"{alg}: {model.eval(var)}")
            
            if values:
                print(f"Variable {var}: {', '.join(values)}")
    else:
        # For scheduling problem (start times)
        for i, var in enumerate(problem_vars):
            values = []
            for alg, model in zip(algorithm_names, models):
                if model:
                    values.append(f"{alg}: {model.eval(var)}")
            
            if values:
                print(f"start_{i}: {', '.join(values)}")


def compare_algorithms(timeout=30):
    """Compare all algorithms on different problem types"""
    print("=== MaxSMT Algorithm Comparison ===\n")
    algorithms = ["core-guided", "ihs", "local-search", "z3-opt"]
    
    problems = [
        ("Scheduling Problem", generate_scheduling_problem()),
        ("Boolean Problem", generate_boolean_problem()),
        ("Integer Problem", generate_integer_problem())
    ]
    
    for problem_name, (hard, soft, weights, problem_vars) in problems:
        print(f"\n=== {problem_name} ===")
        
        results = {}
        successful_models = []
        successful_algs = []
        
        for alg in algorithms:
            sat, model, cost, time = test_algorithm(hard, soft, weights, alg, timeout)
            
            # Store results
            if sat:
                successful_models.append(model)
                successful_algs.append(alg)
                results[alg] = {
                    "sat": True,
                    "time": time,
                    "cost": cost
                }
            else:
                successful_models.append(None)
                results[alg] = {
                    "sat": False,
                    "time": time
                }
        
        # Compare variable assignments across algorithms
        if successful_models:
            compare_models(algorithms, successful_models, problem_vars)
        
        # Compare results
        print("\nResults Summary:")
        print("Algorithm | Success | Time (s) | Cost")
        print("----------+---------+----------+------")
        
        for alg in algorithms:
            if results[alg]["sat"]:
                print(f"{alg:10} | Yes     | {results[alg]['time']:8.4f} | {results[alg]['cost']:6.2f}")
            else:
                print(f"{alg:10} | No      | {results[alg]['time']:8.4f} | --")
        
        # Find the best algorithm for this problem
        successful = [alg for alg in algorithms if results[alg].get("sat", False)]
        
        if successful:
            # Find fastest algorithm
            fastest = min(successful, key=lambda alg: results[alg]["time"])
            
            # Find best solution (lowest cost)
            best_cost = min(successful, key=lambda alg: results[alg]["cost"])
            
            print(f"\nFastest algorithm: {fastest} ({results[fastest]['time']:.4f}s)")
            print(f"Best solution: {best_cost} (cost={results[best_cost]['cost']:.2f})")
        else:
            print("\nNo algorithm found a solution.")


if __name__ == "__main__":
    # Check if timeout specified as command line arg
    timeout = 60 if "--extended" in sys.argv else 30
    
    # Run the comparison
    compare_algorithms(timeout=timeout) 