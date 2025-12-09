"""Examples and test cases for the PBE solver.

This module demonstrates the usage of the PBE solver with different theories.
"""

from .pbe_solver import PBESolver, SynthesisResult
from .expressions import Theory


def test_lia_examples():
    """Test LIA (Linear Integer Arithmetic) synthesis."""
    print("Testing LIA synthesis...")

    # Example 1: Simple addition
    examples = [
        {"x": 1, "y": 2, "output": 3},
        {"x": 3, "y": 4, "output": 7},
        {"x": 5, "y": 1, "output": 6},
    ]

    solver = PBESolver(max_expression_depth=2, timeout=10.0)
    result = solver.synthesize(examples)

    print(f"  Example 1 (x + y): {result}")

    # Example 2: Multiplication
    examples = [
        {"a": 2, "b": 3, "output": 6},
        {"a": 4, "b": 5, "output": 20},
        {"a": 3, "b": 7, "output": 21},
    ]

    result = solver.synthesize(examples)
    print(f"  Example 2 (a * b): {result}")

    # Example 3: More complex expression
    examples = [
        {"x": 5, "output": 15},  # 3 * 5
        {"x": 3, "output": 9},   # 3 * 3
        {"x": 2, "output": 6},   # 3 * 2
    ]

    result = solver.synthesize(examples)
    print(f"  Example 3 (3 * x): {result}")


def test_string_examples():
    """Test String theory synthesis."""
    print("\nTesting String synthesis...")

    # Example 1: String concatenation
    examples = [
        {"s1": "Hello", "s2": "World", "output": "HelloWorld"},
        {"s1": "Hi", "s2": "There", "output": "HiThere"},
        {"s1": "A", "s2": "B", "output": "AB"},
    ]

    solver = PBESolver(max_expression_depth=2, timeout=10.0)
    result = solver.synthesize(examples)

    print(f"  Example 1 (s1 ++ s2): {result}")

    # Example 2: String length
    examples = [
        {"s": "Hello", "output": 5},
        {"s": "Hi", "output": 2},
        {"s": "", "output": 0},
    ]

    result = solver.synthesize(examples)
    print(f"  Example 2 (len(s)): {result}")


def test_bv_examples():
    """Test BitVector theory synthesis."""
    print("\nTesting BitVector synthesis...")

    # Example 1: Bitwise AND
    examples = [
        {"x": 0b1010, "y": 0b1100, "output": 0b1000},  # 10 & 12 = 8
        {"x": 0b1111, "y": 0b0011, "output": 0b0011},  # 15 & 3 = 3
        {"x": 0b0000, "y": 0b1111, "output": 0b0000},  # 0 & 15 = 0
    ]

    solver = PBESolver(max_expression_depth=2, timeout=10.0)
    result = solver.synthesize(examples)

    print(f"  Example 1 (x & y): {result}")

    # Example 2: Bitwise XOR
    examples = [
        {"x": 0b1010, "y": 0b1100, "output": 0b0110},  # 10 ^ 12 = 6
        {"x": 0b1111, "y": 0b0011, "output": 0b1100},  # 15 ^ 3 = 12
        {"x": 0b0000, "y": 0b1111, "output": 0b1111},  # 0 ^ 15 = 15
    ]

    result = solver.synthesize(examples)
    print(f"  Example 2 (x ^ y): {result}")


def test_counterexample_generation():
    """Test counterexample generation."""
    print("\nTesting counterexample generation...")

    solver = PBESolver()

    # Create some simple expressions manually for testing
    from .expressions import var, const, add, mul, Theory

    x = var("x", Theory.LIA)
    y = var("y", Theory.LIA)

    expressions = [
        add(x, y),  # x + y
        mul(x, y),  # x * y
    ]

    examples = [
        {"x": 1, "y": 2, "output": 3},  # Both expressions give 3
    ]

    counterexample = solver.generate_counterexample(expressions, examples)
    print(f"  Counterexample found: {counterexample}")


def test_version_space_operations():
    """Test version space operations."""
    print("\nTesting version space operations...")

    solver = PBESolver()

    # Create some simple expressions manually for testing
    from .expressions import var, const, add, mul, sub, Theory

    x = var("x", Theory.LIA)

    expressions = [
        add(x, const(1, Theory.LIA)),  # x + 1
        mul(x, const(2, Theory.LIA)),  # x * 2
        sub(x, const(1, Theory.LIA)),  # x - 1
    ]

    examples = [
        {"x": 2, "output": 3},  # Only x + 1 gives 3
    ]

    # Create version space
    from .vsa import VersionSpace
    vs = VersionSpace(set(expressions))

    # Create algebra and filter with example
    def expression_generator():
        return [var("x", Theory.LIA), const(0, Theory.LIA), const(1, Theory.LIA), const(2, Theory.LIA),
                add(var("x", Theory.LIA), const(1, Theory.LIA)),
                mul(var("x", Theory.LIA), const(2, Theory.LIA)),
                sub(var("x", Theory.LIA), const(1, Theory.LIA))]

    from .vsa import VSAlgebra
    algebra = VSAlgebra(Theory.LIA, expression_generator)
    filtered_vs = algebra.filter_consistent(vs, examples)

    print(f"  Original version space size: {len(vs)}")
    print(f"  Filtered version space size: {len(filtered_vs)}")
    print(f"  Remaining expressions: {[str(expr) for expr in filtered_vs.expressions]}")


def demonstrate_usage():
    """Demonstrate typical usage of the PBE solver."""
    print("\n=== PBE Solver Usage Demonstration ===")

    # Example: Synthesize a function that adds 5 to its input
    examples = [
        {"input": 10, "output": 15},
        {"input": 20, "output": 25},
        {"input": 5, "output": 10},
        {"input": 0, "output": 5},
    ]

    solver = PBESolver(max_expression_depth=3, timeout=15.0)
    result = solver.synthesize(examples)

    print(f"Synthesis result: {result}")

    # If successful, verify the solution
    if result.success and result.expression:
        print(f"Verifying solution: {solver.verify(result.expression, examples)}")

    # Example with strings: reverse function
    string_examples = [
        {"s": "hello", "output": "olleh"},
        {"s": "abc", "output": "cba"},
        {"s": "a", "output": "a"},
    ]

    # This would require a more sophisticated expression generator
    print("String reversal example (would need more sophisticated synthesis):")
    print(f"  Examples: {string_examples}")
    print("  This would require extending the expression generator for string manipulation")


if __name__ == "__main__":
    test_lia_examples()
    test_string_examples()
    test_bv_examples()
    test_counterexample_generation()
    test_version_space_operations()
    demonstrate_usage()
