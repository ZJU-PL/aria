# AGENTS.md - Developer Guidelines for ARIA

This file provides guidance for AI agents working in this repository.

## Project Overview

ARIA is a library for automated reasoning including SMT solving, model counting, and symbolic computation. It supports Python 3.8+.

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest aria/tests/test_bool_engines.py

# Run a specific test class
pytest aria/tests/test_bool_engines.py::TestBoolEngines

# Run a specific test method
pytest aria/tests/test_bool_engines.py::TestBoolEngines::test_models_sampling_and_reducing

# Run tests with coverage
pytest --cov=aria

# Run tests excluding slow tests
pytest -m "not slow"

# Run a single test using Python
python -m pytest aria/tests/test_bool_engines.py::TestBoolEngines::test_models_sampling_and_reducing -v
```

### Linting & Type Checking

```bash
# Run pylint
pylint aria/

# Run mypy (python_version = 3.9)
mypy aria/
```

### Code Formatting

```bash
# Format with black (line-length = 88) and sort imports with isort
black aria/ && isort aria/
```

## Code Style Guidelines

### General

- Maximum line length: **88 characters** (Black default)
- Use type hints (required - `disallow_untyped_defs = true` in mypy)

### Formatting & Imports

- Use **Black** for formatting and **isort** for imports
- Group imports: standard library → third-party → local

```python
import os
from typing import List, Optional, Tuple
from pysat.solvers import Solver
from aria.bool.maxsat import AnytimeMaxSAT
```

### Type Hints

- Use `Optional[X]` instead of `X | None` (Python 3.9 compatibility)
- Use `List`, `Dict`, `Tuple` from typing (not builtins)

```python
# Good
def solve(self, timeout: int = 300) -> Tuple[bool, Optional[List[int]], int]:
    ...

# Avoid
def solve(self, timeout = 300):
    ...
```

### Naming Conventions

- Classes: `CamelCase` (e.g., `AnytimeMaxSAT`)
- Functions/methods: `snake_case` (e.g., `solve_maxsat`)
- Private methods: prefix with `_` (e.g., `_compute_cost`)

### Docstrings

Use Google-style or NumPy-style. Keep brief and concise.

```python
def solve(self, timeout: int = 300) -> Tuple[bool, Optional[List[int]], int]:
    """Solve MaxSAT problem using core-guided approach.

    Args:
        timeout: Maximum time in seconds.

    Returns:
        Tuple of (success, model, cost).
    """
```

### Test Code

Use the custom `TestCase` from `aria.tests`:

```python
from aria.tests import TestCase, main

class TestBoolEngines(TestCase):
    def test_something(self):
        ...

if __name__ == "__main__":
    main()
```

### Working with LLM-Generated Code

This repository contains LLM-generated code marked with `FIXME` or `TODO`. When modifying such code:

1. Verify the algorithm against academic papers or established implementations
2. Test against known solvers (e.g., PySat's RC2 for MaxSAT)
3. Remove the `FIXME` comment once validated

### Common Patterns

```python
# SAT solver (pysat)
solver = Solver(name='glucose4', bootstrap_with=hard_clauses, incr=True)
solver.add_clause([1, 2, -3])
result = solver.solve()
if result:
    model = solver.get_model()
solver.delete()

# PySMT
x = Symbol("x", INT)
with Solver(name="z3", logic=QF_LIA) as solver:
    solver.add_assertion(x > y)
    if solver.solve():
        model = solver.get_model()
```

## File Organization

```
aria/
├── bool/           # SAT, MaxSAT, CNF
├── smt/            # SMT solving
├── quant/          # Quantified formulas
├── optimization/   # MaxSAT, OMT
├── srk/            # Symbolic reasoning kernel
├── sampling/       # Model sampling
├── tests/          # Main test suite
└── cli/            # Command-line interfaces
```

## Key Dependencies

- **PySMT**: Formula manipulation and SMT interface
- **z3-solver**: Z3 SMT solver
- **python-sat**: SAT solver interface (glucose, cadical, etc.)

## Environment Variables

- `ARIA_DEBUG`: Set to "true" to enable debug mode
