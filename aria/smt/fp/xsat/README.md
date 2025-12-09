# XSat - Floating-Point Satisfiability Solver

XSat is a satisfiability solver for floating-point constraints using optimization-based techniques. It translates SMT-LIB2 floating-point formulas into C code and uses numerical optimization to find satisfying assignments.

## Features

- **Floating-Point Support**: Handles Float32 and Float64 IEEE 754 floating-point types
- **Multiple Precision Modes**:
  - `R_square`: Fast approximation using squared error
  - `R_ulp`: High precision using ULP (Unit in Last Place) distance
  - `R_verify`: Exact verification mode
- **Optimization-Based**: Uses scipy's basinhopping algorithm for global optimization
- **Parallel Processing**: Supports multi-core execution (?)

## Requirements

- Python 3.8+
- Z3 Theorem Prover
- NumPy, SciPy, SymPy
- GCC/Clang compiler

## Installation

1. Activate the virtual environment:
   ```bash
   source /path/to/aria/test_venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install sympy scipy
   ```

## Usage

### Code Generation

Generate C code from an SMT-LIB2 file:

```bash
python xsat_gen.py input.smt2
```

This creates:
- `build/foo.c`: Generated C constraint function
- `build/foo.symbolTable`: Variable type information

### Compilation

Compile the generated code with different precision modes:

```bash
make compile_square    # Fast approximation
make compile_ulp       # High precision
make compile_verify    # Verification mode
```

### Solving

Run the solver:

```bash
make solve
```

Or directly:

```bash
python xsat.py
```

## Command Line Options

### xsat_gen.py
- `smt2_file`: Input SMT-LIB2 file
- `--method`: Optimization method (powell, bfgs, etc.)
- `--showResult`: Display optimization results
- `--verify`: Enable model verification

### xsat.py
- `--multi`: Enable multi-processing
- `--stepSize`: Optimization step size
- `--round2_threshold`: Threshold for round 2 optimization
- `--verify`: Verify solutions with Z3

## Example

```bash
# Generate code from SMT2 file
python xsat_gen.py example.smt2

# Compile with ULP precision
make compile_ulp

# Solve the constraints
python xsat.py
```

## Architecture

XSat works in multiple rounds:
1. **Round 1**: Fast optimization with squared error
2. **Round 2**: Refined optimization with ULP scaling
3. **Round 3**: Final verification and refinement

The solver generates shared libraries that are called from Python during optimization.

## Python 3 Migration

This version has been migrated from Python 2 to Python 3 with the following changes:
- Print statements converted to print functions
- Dictionary iteration updated to use `.items()`
- Import statements modernized (cPickle → pickle, reload → importlib.reload)
- Compilation flags updated for Python 3

## Contributors

- Zhoulai Fu
- Zhendong Su
