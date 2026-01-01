"""
Example usage of SYMBA for symbolic optimization.

This script demonstrates how to use the SYMBA algorithm for optimizing
objective functions in linear real arithmetic using SMT solvers.
"""

import os
import sys

import z3

# Add the parent directory to the path to import aria modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aria.symabs.symba import SYMBA, MultiSYMBA


def example_1_resource_allocation():
    """Example 1: Resource allocation problem."""
    print("Example 1: Resource Allocation")
    print("-" * 40)

    # Scenario: Allocate resources to two projects
    # Project A: cost 2, benefit 3
    # Project B: cost 1, benefit 2
    # Budget: 10

    # Variables: amount allocated to each project
    x = z3.Int("x")  # allocation to project A
    y = z3.Int("y")  # allocation to project B

    # Constraints
    constraints = z3.And(
        x >= 0,
        y >= 0,  # non-negative allocations
        2 * x + 1 * y <= 10,  # budget constraint
    )

    # Objective: maximize total benefit
    objective = 3 * x + 2 * y

    print(f"Constraints: {constraints}")
    print(f"Objective: maximize {objective}")

    # Create and run SYMBA
    symba = SYMBA(constraints, [objective])
    state = symba.optimize()

    print(f"\nOptimization completed in {symba.stats['total_time']:.3f}s")
    print(f"SMT queries: {symba.stats['smt_queries']}")
    print(f"Rules applied: {symba.stats['rules_applied']}")

    # Display results
    optimal_values = symba.get_optimal_values()
    print(f"\nOptimal value: {optimal_values[objective]}")

    if state.M:
        print("\nOptimal solutions found:")
        for i, model in enumerate(state.M):
            x_val = model.eval(x).as_long()
            y_val = model.eval(y).as_long()
            obj_val = model.eval(objective).as_long()
            print(f"  Solution {i+1}: x={x_val}, y={y_val}, benefit={obj_val}")

    print()


def example_2_production_optimization():
    """Example 2: Production optimization with multiple objectives."""
    print("Example 2: Production Optimization")
    print("-" * 40)

    # Scenario: Factory producing two products
    # Product A: profit 5, resource usage 3
    # Product B: profit 3, resource usage 2
    # Resource limit: 20

    # Variables
    a = z3.Int("a")  # units of product A
    b = z3.Int("b")  # units of product B

    # Constraints
    constraints = z3.And(
        a >= 0,
        b >= 0,  # non-negative production
        3 * a + 2 * b <= 20,  # resource constraint
    )

    # Multi-objective: maximize profit and minimize resource usage
    profit = 5 * a + 3 * b
    resource_usage = 3 * a + 2 * b

    print(f"Constraints: {constraints}")
    print(f"Objectives: maximize {profit}, minimize {resource_usage}")

    # Create and run MultiSYMBA
    multi_symba = MultiSYMBA(constraints, [profit, resource_usage])
    state = multi_symba.optimize()

    print(f"\nOptimization completed in {multi_symba.symba.stats['total_time']:.3f}s")
    print(f"SMT queries: {multi_symba.symba.stats['smt_queries']}")

    # Display Pareto front
    pareto_values = multi_symba.get_pareto_values()
    print(f"\nPareto front ({len(pareto_values)} solutions):")
    for i, (prof, res) in enumerate(pareto_values):
        print(f"  Solution {i+1}: profit={prof}, resource_usage={res}")

    if state.M:
        print("\nDetailed solutions:")
        for i, model in enumerate(state.M):
            a_val = model.eval(a).as_long()
            b_val = model.eval(b).as_long()
            prof_val = model.eval(profit).as_long()
            res_val = model.eval(resource_usage).as_long()
            is_pareto = model in multi_symba.get_pareto_front()
            print(
                f"  Solution {i+1}: a={a_val}, b={b_val}, profit={prof_val}, "
                f"resource={res_val} {'(Pareto optimal)' if is_pareto else ''}"
            )

    print()


def example_3_bounded_optimization():
    """Example 3: Bounded optimization problem."""
    print("Example 3: Bounded Optimization")
    print("-" * 40)

    # Variables
    x = z3.Int("x")
    y = z3.Int("y")

    # Constraints forming a bounded region
    constraints = z3.And(x >= 0, y >= 0, x <= 5, y <= 5, x + y >= 3)

    # Objective: maximize x * y
    objective = x * y

    print(f"Constraints: {constraints}")
    print(f"Objective: maximize {objective}")

    # Create and run SYMBA
    symba = SYMBA(constraints, [objective])
    state = symba.optimize()

    print(f"\nOptimization completed in {symba.stats['total_time']:.3f}s")
    print(f"SMT queries: {symba.stats['smt_queries']}")

    # Display results
    optimal_values = symba.get_optimal_values()
    print(f"\nOptimal value: {optimal_values[objective]}")

    if state.M:
        print("\nOptimal solutions found:")
        for i, model in enumerate(state.M):
            x_val = model.eval(x).as_long()
            y_val = model.eval(y).as_long()
            obj_val = model.eval(objective).as_long()
            print(f"  Solution {i+1}: x={x_val}, y={y_val}, product={obj_val}")

    print()


def run_examples():
    """Run all examples."""
    print("SYMBA Examples")
    print("=" * 50)

    example_1_resource_allocation()
    example_2_production_optimization()
    example_3_bounded_optimization()

    print("=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    run_examples()
