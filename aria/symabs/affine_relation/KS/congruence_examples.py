from __future__ import annotations

import z3

from aria.symabs.affine_relation.KS.congruence_abstraction import congruent_closure
from aria.symabs.affine_relation.KS.loop_analysis import (
    analyze_python_loop,
    example_parity_loop,
    example_bit_counting,
)


def parity_example(width: int = 4) -> None:
    """Example from the paper: parity computation with enhanced congruence analysis."""
    # Simple parity program fragment: x' is x shifted; p' toggles.
    # We only model a single step to keep it tiny.
    xs = [z3.Bool(f"x{i}") for i in range(width)]
    ps = [z3.Bool(f"p{i}") for i in range(width)]
    # Toy relation: p1 = x0 XOR p0, p2 = x1 XOR p1, ... (ripple parity)
    cnstrs = []
    for i in range(1, width):
        cnstrs.append(ps[i] == z3.Xor(xs[i - 1], ps[i - 1]))
    phi = z3.And(*cnstrs) if cnstrs else z3.BoolVal(True)

    sys = congruent_closure(phi, xs + ps, modulus=1 << 1)  # modulo 2
    print("Enhanced congruence system (mod 2):", sys)
    print(f"System has {sys.num_rows} rows and width {sys.width}")


def bit_counting_example() -> None:
    """Example: bit counting with congruence analysis."""
    # Simple bit counting: count the number of 1s in 4 bits
    x_bits = [z3.Bool(f"x{i}") for i in range(4)]
    total = sum(z3.If(b, 1, 0) for b in x_bits)

    # For this simple case, we expect total to be congruent to 0, 1, 2, 3, or 4 mod 8
    # But let's create a constraint that total is even (for demonstration)
    phi = total % 2 == 0  # Total is even

    # Analyze with different moduli
    for mod in [2, 4, 8]:
        sys = congruent_closure(phi, x_bits, modulus=mod)
        print(f"Congruence system (mod {mod}): {sys}")
        print(f"  Rows: {sys.num_rows}, Width: {sys.width}")


def loop_analysis_example() -> None:
    """Demonstrate AST-based loop analysis."""
    print("Analyzing parity loop with AST...")
    constraints, variables = analyze_python_loop(example_parity_loop)
    print(f"Found {len(constraints)} constraints and {len(variables)} variables")

    print("\nAnalyzing bit counting loop with AST...")
    constraints, variables = analyze_python_loop(example_bit_counting)
    print(f"Found {len(constraints)} constraints and {len(variables)} variables")


def complex_loop_example() -> None:
    """More complex loop example showing enhanced congruence analysis."""

    # Example from typical bit-twiddling: reversing bits in a word
    def bit_reverse():
        x = 42  # example input
        result = 0
        for i in range(8):  # 8-bit example
            # Extract bit i and place it at position (7-i)
            bit = (x >> i) & 1
            result = result | (bit << (7 - i))
        return result

    # Analyze the function
    constraints, variables = analyze_python_loop(bit_reverse)
    print("Bit reversal loop analysis:")
    print(f"Found {len(constraints)} constraints and {len(variables)} variables")

    # The bit reversal should create specific congruence relationships
    # For example, result should be congruent to the bit-reversed version of x


if __name__ == "__main__":
    print("=== Enhanced Congruence Analysis Examples ===\n")

    print("1. Basic Parity Example:")
    parity_example()

    print("\n2. Bit Counting Example:")
    bit_counting_example()

    print("\n3. Loop Analysis Examples:")
    loop_analysis_example()

    print("\n4. Complex Loop Example:")
    complex_loop_example()
