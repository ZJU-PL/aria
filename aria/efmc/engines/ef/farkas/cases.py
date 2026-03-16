"""
Test cases for Farkas-based invariant inference.
"""

import z3
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.farkas.farkas_prover import FarkasProver


def create_transition_system(variables, prime_vars, init, trans, post):
    """Create a transition system from components."""
    sts = TransitionSystem()
    sts.variables = variables
    sts.prime_variables = prime_vars
    sts.init = init
    sts.trans = trans
    sts.post = post
    sts.all_variables = variables + prime_vars
    sts.has_int = any(v.sort() == z3.IntSort() for v in variables)
    sts.has_real = any(v.sort() == z3.RealSort() for v in variables)
    sts.has_bv = any(z3.is_bv_sort(v.sort()) for v in variables)
    sts.has_bool = any(v.sort() == z3.BoolSort() for v in variables)
    sts.initialized = True
    return sts


def create_test_cases():
    """Create test cases for Farkas prover."""
    test_cases = []

    # Simple Counter Program
    x, x_prime = z3.Int("x"), z3.Int("x_prime")
    sts1 = create_transition_system([x], [x_prime], x == 0, x_prime == x + 1, x >= 0)
    test_cases.append(("Simple Counter Program", sts1))

    # Two-Variable Counter
    x, y = z3.Int("x"), z3.Int("y")
    x_prime, y_prime = z3.Int("x_prime"), z3.Int("y_prime")
    sts2 = create_transition_system(
        [x, y],
        [x_prime, y_prime],
        z3.And(x == 0, y == 10),
        z3.And(x_prime == x + 1, y_prime == y),
        x <= y,
    )
    test_cases.append(("Two-Variable Counter", sts2))

    # Difference Bounds
    x, y = z3.Int("x"), z3.Int("y")
    x_prime, y_prime = z3.Int("x_prime"), z3.Int("y_prime")
    sts3 = create_transition_system(
        [x, y],
        [x_prime, y_prime],
        z3.And(x == y, y == 0),
        z3.And(x_prime == x + 1, y_prime == y),
        x - y <= 5,
    )
    test_cases.append(("Difference Bounds", sts3))

    # Multi-Path Loop
    x, x_prime = z3.Int("x"), z3.Int("x_prime")
    sts4 = create_transition_system(
        [x],
        [x_prime],
        x == 0,
        z3.Or(z3.And(x < 10, x_prime == x + 1), z3.And(x >= 10, x_prime == x)),
        x >= 0,
    )
    test_cases.append(("Multi-Path Loop", sts4))

    # Array Bounds
    i, n = z3.Int("i"), z3.Int("n")
    i_prime, n_prime = z3.Int("i_prime"), z3.Int("n_prime")
    sts5 = create_transition_system(
        [i, n],
        [i_prime, n_prime],
        z3.And(i == 0, n >= 0),
        z3.And(i < n, i_prime == i + 1, n_prime == n),
        n >= i,
    )
    test_cases.append(("Array Bounds", sts5))

    # Multiple Invariants
    x, y = z3.Int("x"), z3.Int("y")
    x_prime, y_prime = z3.Int("x_prime"), z3.Int("y_prime")
    sts6 = create_transition_system(
        [x, y],
        [x_prime, y_prime],
        z3.And(x == 0, y == 0),
        z3.And(x_prime == x + 1, y_prime == y + 2),
        y == 2 * x,
    )
    test_cases.append(("Multiple Invariants", sts6))

    return test_cases


def run_farkas_prover_test(name, sts):
    """Run Farkas prover on a single test case."""
    print(f"{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    var_names = [str(v) for v in sts.variables]
    print(f"Variables: {var_names}")
    print(f"Init: {sts.init}")
    print(f"Trans: {sts.trans}")
    print(f"Post: {sts.post}")

    prover = FarkasProver(sts, num_templates=3, verbose=False, validate_invariant=True)
    result = prover.solve(timeout=30)

    print(f"\nResult: {result}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Is Unknown: {result.is_unknown}")

    if result.invariant:
        print(f"Invariant: {result.invariant}")
    else:
        print("No invariant found")

    stats = prover.get_statistics()
    print(f"Statistics: {stats}")

    return result


def demo_cases():
    """Run all test cases and summarize results."""
    print("Farkas-based Invariant Inference Test Cases")
    print("=" * 60)

    test_cases = create_test_cases()
    results = {}

    for name, sts in test_cases:
        print()
        result = run_farkas_prover_test(name, sts)

        if result.is_safe:
            status = "SAFE"
        elif result.is_unknown:
            status = "UNKNOWN"
        else:
            status = "ERROR"

        results[name] = status

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, status in results.items():
        print(f"{name:<30} : {status}")

    safe_count = sum(1 for s in results.values() if s == "SAFE")
    unknown_count = sum(1 for s in results.values() if s == "UNKNOWN")
    error_count = sum(1 for s in results.values() if s == "ERROR")

    print(f"\nTotal tests: {len(results)}")
    print(f"Safe: {safe_count}, Unknown: {unknown_count}, Errors: {error_count}")


if __name__ == "__main__":
    demo_cases()
