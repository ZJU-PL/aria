# FOSSIL: Framework for Synthesis of Inductive Lemmas

Adapted from https://github.com/muraliadithya/FOSSIL

FOSSIL is a framework for automated synthesis of inductive lemmas to aid in program verification. It combines natural proofs with syntax-guided synthesis (SyGuS) to automatically discover lemmas that can prove program correctness properties, particularly for data structures involving recursive definitions.

## Overview

FOSSIL consists of three main components:

1. **Natural Proofs Solver** (`naturalproofs/`) - A proof system for reasoning about recursive data structures
2. **Lemma Synthesis Engine** (`lemsynth/`) - Automated synthesis of inductive lemmas using SyGuS
3. **Benchmark Suite** (`benchmark-suite/`) - Collection of verification problems for data structures

## Components

### Natural Proofs (`naturalproofs/`)

The natural proofs component provides a proof system specifically designed for reasoning about recursive data structures like lists, trees, and heaps. It uses finite model finding and instantiation strategies to prove properties.

**Key Features:**
- Finite model-based proof search
- Multiple instantiation strategies (bounded depth, stratified, lean)
- Support for recursive definitions and axioms
- Integration with Z3 SMT solver

**Main Classes:**
- `NPSolver` - Main solver class for natural proofs
- `NPSolution` - Represents solutions/results from the solver
- `AnnotatedContext` - Stores vocabulary, recursive definitions, and axioms

### Lemma Synthesis (`lemsynth/`)

The lemma synthesis engine automatically discovers inductive lemmas that can help prove verification conditions. It uses syntax-guided synthesis to generate candidate lemmas and validates them using the natural proofs solver.

**Key Features:**
- SyGuS-based lemma synthesis
- Integration with CVC4/cvc5 synthesis solvers
- Grammar-guided lemma generation
- Counterexample-guided refinement

**Main Modules:**
- `lemma_synthesis.py` - Core synthesis algorithms
- `lemsynth_engine.py` - High-level synthesis interface
- `grammar_utils.py` - Grammar handling utilities
- `true_models.py` - Model generation and validation

### Benchmark Suite (`benchmark-suite/`)

A comprehensive collection of verification problems for common data structures and their properties.

**Categories:**
- **Binary Search Trees** - BST invariants and operations
- **Linked Lists** - List properties, reachability, segments
- **Trees** - Tree structures, reachability, parent-child relationships
- **Heaps** - Max-heap properties and tree structures
- **Reachability** - Graph reachability problems

## Requirements

- Python 3.5 or above
- [Z3Py](https://pypi.org/project/z3-solver/) - SMT solver with Python bindings
- CVC4 or cvc5 (for lemma synthesis)

## Installation

1. Install Z3Py:
   ```bash
   pip3 install z3-solver
   ```

2. Install CVC4 or cvc5 for synthesis capabilities

3. Add the naturalproofs directory to your PYTHONPATH:
   ```bash
   export PYTHONPATH="/path/to/aria/fossil/naturalproofs":$PYTHONPATH
   ```

## Usage

### Basic Natural Proofs Example

```python
from aria.fossil.naturalproofs.prover import NPSolver
from aria.fossil.naturalproofs.uct import fgsort, boolsort
from aria.fossil.naturalproofs.decl_api import Var, Function, RecFunction, AddRecDefinition
from aria.fossil.naturalproofs.pfp import make_pfp_formula

# Declare variables and functions
x = Var('x', fgsort)
nil = Const('nil', fgsort)
next = Function('next', fgsort, fgsort)
list_pred = RecFunction('list', fgsort, boolsort)

# Add recursive definition
AddRecDefinition(list_pred, x, 
    If(x == nil, True, list_pred(next(x))))

# Create solver and prove property
np_solver = NPSolver()
goal = Implies(list_pred(x), list_pred(next(x)))
solution = np_solver.solve(make_pfp_formula(goal))

if not solution.if_sat:
    print('Goal is valid')
else:
    print('Goal is invalid')
```

### Lemma Synthesis Example

```python
from aria.fossil.lemsynth.lemsynth_engine import solveProblem

# Define lemma grammar arguments and terms
lemma_grammar_args = [x, nil]
lemma_grammar_terms = {x, nil, next(x), next(next(x))}

# Synthesis goal
goal = Implies(list_pred(x), some_property(x))

# Run synthesis
solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, 
            'list-example', grammar_string)
```

### Running Benchmarks

```python
# Run a specific benchmark
python aria/fossil/benchmark-suite/bst-tree.py
```

## Configuration Options

### Natural Proofs Solver Options

- **Instantiation Modes:**
  - `bounded_depth` - Limited depth instantiation
  - `fixed_depth` - Fixed depth instantiation  
  - `infinite_depth` - Unbounded instantiation
  - `manual_instantiation` - User-specified terms
  - `lean_instantiation` - Selective instantiation
  - `depth_one_stratified_instantiation` - Stratified approach

- **Depth Control:** Set maximum instantiation depth
- **Term Selection:** Control which terms to use for instantiation

### Lemma Synthesis Options

- **Synthesis Solver:** Choose between CVC4, cvc5, or other SyGuS solvers
- **Grammar:** Specify synthesis grammar for lemma generation
- **Timeout:** Set time limits for synthesis attempts
- **Logging:** Control output verbosity and logging

## Architecture

```
aria/fossil/
├── naturalproofs/          # Natural proofs solver
│   ├── prover.py          # Main solver implementation
│   ├── decl_api.py        # Declaration API
│   ├── uct.py             # Uninterpreted types
│   ├── pfp.py             # Proof-relevant formula processing
│   └── extensions/        # Solver extensions
├── lemsynth/              # Lemma synthesis engine
│   ├── lemma_synthesis.py # Core synthesis algorithms
│   ├── lemsynth_engine.py # High-level interface
│   └── utils.py           # Synthesis utilities
└── benchmark-suite/       # Verification benchmarks
    ├── bst-*.py          # Binary search tree problems
    ├── list-*.py         # Linked list problems
    └── tree-*.py         # Tree structure problems
```

## Key Concepts

### Natural Proofs
A proof methodology that uses finite models to reason about infinite structures. The approach is particularly effective for recursive data structures where traditional proof methods may struggle.

### Syntax-Guided Synthesis (SyGuS)
A framework for automatically synthesizing programs (in this case, lemmas) that satisfy given specifications while conforming to syntactic constraints defined by a grammar.

### Inductive Lemmas
Auxiliary properties that help prove main verification goals. These lemmas often capture invariants or intermediate properties of recursive data structures.

## Research Background

FOSSIL implements techniques from several research papers in automated verification and synthesis:

- Natural proofs for reasoning about data structures
- Syntax-guided synthesis for lemma discovery
- Finite model finding for infinite structures
- Counterexample-guided refinement for synthesis

## Contributing

When adding new benchmarks or extending functionality:

1. Follow the existing code structure and naming conventions
2. Add appropriate documentation and examples
3. Test with multiple instantiation strategies
4. Include both positive and negative test cases

## Troubleshooting

**Common Issues:**

1. **Z3 Import Errors:** Ensure Z3Py is properly installed and in Python path
2. **Synthesis Timeouts:** Increase timeout limits or simplify grammar
3. **Memory Issues:** Reduce instantiation depth or use lean instantiation
4. **CVC4/cvc5 Not Found:** Ensure synthesis solver is installed and in PATH

For more detailed examples, see the `tests/` directories and benchmark files.