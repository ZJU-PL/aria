# Version Space Algebra (VSA) and Programming by Example (PBE) Implementation

This document describes the implementation of Version Space Algebra and Programming by Example solver in the Aria synthesis module.

## Overview

The implementation consists of two main components:

1. **Version Space Algebra (VSA)**: An algebraic framework for representing and manipulating sets of programs that are consistent with observed examples.
2. **Programming by Example (PBE) Solver**: A synthesis tool that uses VSA to generate programs from input-output examples.

## Architecture

```
aria/synthesis/
├── vsa/                          # Version Space Algebra
│   ├── __init__.py              # Module exports
│   ├── expressions.py           # Expression types (LIA, BV, String)
│   └── vsa.py                   # Core VSA implementation
└── pbe/                         # Programming by Example solver
    ├── __init__.py              # Module exports
    ├── expression_generators.py # Expression generation for theories
    ├── pbe_solver.py            # Main PBE solver
    └── examples.py              # Usage examples and tests
```

## Version Space Algebra (VSA)

### Core Concepts

- **Version Space**: A set of expressions (programs) that are consistent with given examples
- **Algebraic Operations**: Join (union), Meet (intersection), and filtering operations
- **Theories Supported**: Linear Integer Arithmetic (LIA), BitVectors (BV), Strings

### Key Classes

- `VersionSpace`: Represents a set of expressions with algebraic operations
- `VSAlgebra`: Provides algebraic operations on version spaces
- `Expression`: Abstract base class for all expressions
- `Variable`, `Constant`, `BinaryExpr`, `UnaryExpr`: Concrete expression types

### Expression Types

#### Linear Integer Arithmetic (LIA)
```python
# Variables and constants
x = var("x", Theory.LIA)
const_5 = const(5, Theory.LIA)

# Operations
expr = add(x, const_5)  # x + 5
expr = mul(x, const(2, Theory.LIA))  # x * 2
expr = eq(x, const(10, Theory.LIA))  # x == 10
```

#### BitVectors (BV)
```python
# Bitwise operations
expr = bv_and(x, const(0xFF, Theory.BV))  # x & 0xFF
expr = BinaryExpr(x, BinaryOp.BVXOR, y)   # x ^ y
```

#### Strings
```python
# String operations
expr = concat(s1, s2)  # s1 ++ s2
expr = length(s)       # len(s)
```

## Programming by Example (PBE) Solver

### Usage Example

```python
from aria.synthesis.pbe import PBESolver

# Create solver with configuration
solver = PBESolver(max_expression_depth=3, timeout=30.0)

# Define examples
examples = [
    {"x": 1, "y": 2, "output": 3},
    {"x": 3, "y": 4, "output": 7},
    {"x": 5, "y": 1, "output": 6},
]

# Synthesize program
result = solver.synthesize(examples)

if result.success and result.expression:
    print(f"Found program: {result.expression}")
    # Verify the solution
    print(f"Verification: {solver.verify(result.expression, examples)}")
```

### Key Features

- **Multi-theory Support**: LIA, BV, and String theories
- **Counterexample Generation**: Automatically finds distinguishing examples
- **Timeout and Depth Control**: Configurable search parameters
- **Version Space Management**: Efficient representation of program spaces

### Expression Generation

The solver automatically generates expressions for each theory:

- **LIA**: Arithmetic operations, comparisons, constants
- **BV**: Bitwise operations, shifts, masks
- **String**: Concatenation, length, equality

## Implementation Details

### Version Space Operations

```python
# Create version spaces
vs1 = VersionSpace({expr1, expr2})
vs2 = VersionSpace({expr3, expr4})

# Algebraic operations
union_vs = vs1.union(vs2)           # Join
intersect_vs = vs1.intersect(vs2)   # Meet
filtered_vs = algebra.filter_consistent(vs, examples)
```

### Counterexample Generation

The solver uses heuristic methods to find counterexamples that distinguish between expressions in the version space:

```python
counterexample = algebra.find_counterexample(version_space, examples)
```

## Testing

Run the basic structure test:
```bash
python3 test_simple.py
```

For full functionality testing (requires dependencies):
```bash
python3 test_vsa_pbe.py
```

## Dependencies

The implementation requires:
- Python 3.6+
- (Optional) pysat for full aria functionality

## Future Enhancements

1. **Advanced Expression Types**: Conditional expressions, loops, function calls
2. **Performance Optimizations**: Caching, parallel processing
3. **Integration**: Connect with existing Aria SMT solvers
4. **Learning**: Machine learning-based expression ranking
5. **Interactive Mode**: Step-by-step synthesis with user guidance

## References

- Mitchell, T. M. (1982). Generalization as search. Artificial Intelligence
- Lau, T., et al. (2003). Programming by demonstration using version space algebra
- Gulwani, S. (2011). Automating string processing in spreadsheets using input-output examples

## Authors

- Version Space Algebra implementation for Aria
- Part of the Aria automated reasoning toolkit
