# Floating-Point SMT

This package contains ARIA's floating-point solving utilities for
quantifier-free floating-point problems and mixed array/bit-vector/floating-point
fragments.

## Main Modules

- `qffp_solver.py`: `QFFPSolver` for `QF_FP` formulas. The solver applies Z3
  simplification and `fpa2bv`, then either:
  - bit-blasts the resulting `QF_BV` formula and dispatches to PySAT, or
  - falls back to `z3.SolverFor("QF_FP")` when the formula is not purely
    propositional after preprocessing.
- `qfaufbvfp_solver.py`: `QFAUFBVFPSolver` for array/uninterpreted
  function/bit-vector/floating-point combinations. It uses a similar tactic
  pipeline, with Ackermannization and optional SAT solving after reduction to a
  propositional bit-vector problem.

## Subdirectories

- `xsat/`: experimental XSAT-related floating-point artifacts and examples.
- `smt2coral/`: utilities for translating SMT-LIB floating-point inputs into
  Coral-oriented formats and counting supported constructs.

## Typical Usage

```python
from aria.smt.fp.qffp_solver import QFFPSolver

solver = QFFPSolver()
result = solver.solve_smt_file("example.smt2")
print(result)
```

For mixed AUFBVFP inputs:

```python
from aria.smt.fp.qfaufbvfp_solver import QFAUFBVFPSolver

solver = QFAUFBVFPSolver()
result = solver.solve_smt_file("example.smt2")
print(result)
```

## Notes

- Both solvers depend on Z3 parsing and tactic support.
- The default SAT backend is a PySAT engine (`mgh`).
- `convert_fp_to_bv(...)` in `qfaufbvfp_solver.py` can be used to dump
  preprocessed floating-point formulas as `QF_BV` SMT-LIB when the reduction is
  successful.
