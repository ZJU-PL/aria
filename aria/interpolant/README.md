# Interpolant Generation

Craig interpolant computation for SMT formulas.

## What is an Interpolant?

Given formulas A and B where A ∧ B is unsatisfiable, a Craig interpolant is a formula P such that:
- A implies P (A ⊨ P)
- P implies B is unsatisfiable (P ∧ B ⊨ ⊥)
- P uses only symbols common to A and B

## Components

- `pysmt_interpolant.py`: PySMT-based interpolation
- `smtinterpol_interpolant.py`: SMTInterpol-based interpolation
- `cvc5_interpolant.py`: CVC5-based interpolation

## Usage

```python
from aria.interpolant import pysmt_interpolant

# Compute interpolant between A and B (where A ∧ B is unsatisfiable)
interpolant = pysmt_interpolant.get_interpolant(A, B)
```

## Applications

- Program verification
- Predicate abstraction
- Refinement in CEGAR loops
- Model checking
