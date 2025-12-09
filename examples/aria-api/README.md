#ARIA API Examples

This directory contains concise examples demonstrating the key functionalities of aria. Since aria doesn't have an abstraction layer for SMT objects, these examples use Z3's Python API objects directly while showcasing aria's much richer functionalities.

## Examples Overview

### 1. AllSMT (`01_allsmt_example.py`)
- **Purpose**: Enumerate all satisfying models of SMT formulas
- **Key Features**:
  - Multiple solver backends (Z3, PySMT, MathSAT)
  - Support for Boolean, integer, real, and mixed formulas
  - Model limit control for infinite spaces
- **Usage**: `python 01_allsmt_example.py`

### 2. Model Sampling (`02_sampling_example.py`)
- **Purpose**: Advanced sampling techniques for solution space exploration
- **Key Features**:
  - Multiple sampling methods (enumeration, MCMC, region-based, hash-based)
  - Support for various logics (QF_BOOL, QF_LRA, QF_LIA, QF_NRA)
  - Custom sampler creation
- **Usage**: `python 02_sampling_example.py`

### 3. Unsat Core (`03_unsat_core_example.py`)
- **Purpose**: Compute unsatisfiable cores and minimal unsatisfiable subsets (MUS)
- **Key Features**:
  - Basic unsat core computation
  - Named constraints for debugging
  - Multiple MUS enumeration
  - Different algorithms comparison
- **Usage**: `python 03_unsat_core_example.py`

### 4. Backbone (`04_backbone_example.py`)
- **Purpose**: Compute backbone literals (literals true in all models)
- **Key Features**:
  - Model enumeration and sequence checking algorithms
  - Support for Boolean and arithmetic formulas
- **Usage**: `python 04_backbone_example.py`

### 5. Quantifier Elimination (`05_quantifier_elimination_example.py`)
- **Purpose**: Eliminate quantifiers from formulas
- **Key Features**:
  - Existential quantifier elimination
  - Linear arithmetic support
  - Comparison with Z3's built-in QE
- **Usage**: `python 05_quantifier_elimination_example.py`

### 6. Abduction (`06_abduction_example.py`)
- **Purpose**: Find explanations/hypotheses for implications
- **Key Features**:
  - QE-based abduction
  - Dillig's abduction algorithm
  - Method comparison
- **Usage**: `python 06_abduction_example.py`

### 7. Model Counting (`07_counting_example.py`)
- **Purpose**: Count the number of satisfying models
- **Key Features**:
  - Boolean model counting with different methods
  - General model counter for various theories
- **Usage**: `python 07_counting_example.py`

### 8. Interpolation (`08_interpolation_example.py`)
- **Purpose**: Compute interpolants between formulas
- **Key Features**:
  - Boolean interpolants using core-based approach
  - PySMT-based interpolation
  - Arithmetic interpolation
- **Usage**: `python 08_interpolation_example.py`

### 9. MaxSMT (`09_maxsmt_example.py`)
- **Purpose**: Solve Maximum Satisfiability problems
- **Key Features**:
  - Multiple algorithms (core-guided, IHS, Z3-opt, local search)
  - Weighted soft constraints
  - Scheduling and optimization examples
- **Usage**: `python 09_maxsmt_example.py`

### 10. Automata (`10_automata_example.py`)
- **Purpose**: Work with finite and symbolic automata
- **Key Features**:
  - Deterministic and nondeterministic finite automata
  - Symbolic finite automata over integers and strings
  - Automata operations (complement, intersection, minimization)
- **Usage**: `python 10_automata_example.py`

### 11. SyGuS (`11_sygus_example.py`)
- **Purpose**: Syntax-guided synthesis of functions
- **Key Features**:
  - Programming by example (PBE) for strings
  - Function synthesis from logical constraints
  - Invariant synthesis for verification
- **Usage**: `python 11_sygus_example.py`

### 12. Unification (`12_unification_example.py`)
- **Purpose**: Term unification and pattern matching
- **Key Features**:
  - Z3 term unification with substitutions
  - Pattern matching and anti-unification
  - Logic variable unification
- **Usage**: `python 12_unification_example.py`

## Running All Examples

To run all examples:
```bash
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

## Key Takeaways

- **Rich Functionality**: aria provides much more advanced features than basic Z3 usage
- **Multiple Algorithms**: Most functionalities offer multiple algorithms for different use cases
- **Unified Interface**: Consistent API across different solver backends
- **Z3 Compatibility**: All examples use Z3 expressions as input
- **Practical Applications**: Examples demonstrate real-world use cases like debugging, optimization, and analysis

## Dependencies

These examples require:
- Python 3.7+
- Z3 Python bindings
- aria (installed locally)

Some advanced features may require additional solvers or libraries that aria can interface with.
