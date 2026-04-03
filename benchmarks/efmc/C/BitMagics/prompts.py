

"""
Generate simple single-loop functions implementing cryptographic algorithms or bit manipulation operations. Requirements

1. Loop structure: Use only single, non-nested loops (while or for loops)

2. Data types: Use only integer scalar variables - no Booleans, arrays, pointers, or complex data structures

3. Annotations: Include:
  - Preconditions using assume() before the loop
  - Postconditions using assert() after the loop
  - The postcondition should express the mathematical/cryptographic property the loop establishes

4. Complexity: The loop invariants needed to help verify these functions should be non-trivial and require bit-level reasoning. Examples include:
  - Bit rotation/shifting with modular arithmetic
  - Bitwise parity/checksum computations
  - Modular exponentiation
  - GCD computation with bit tricks
  - Hash function components (mixing, avalanche)
  - ...

5. Output format: Provide the C code  with assume() and assert() statements
"""
