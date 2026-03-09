# ARIA FF Architecture

The finite-field SMT package is split into three subpackages:

- `aria.smt.ff.frontend`
  - `ff_parser.py`: SMT-LIB finite-field parser
  - `ff_preprocess.py`: normalization and derived-fact rewrites
- `aria.smt.ff.solvers`
  - `ff_bv_solver.py`: bit-vector backend
  - `ff_bv_solver2.py`: BV/Int bridge backend
  - `ff_int_solver.py`: integer backend
  - `ff_perf_solver.py`: performance backend (adaptive scheduling + kernels +
    partition-driven CEGAR refinement with modulo-aware cuts and local algebra)
  - `ff_solver.py`: automatic backend selector (`FFAutoSolver`)
- `aria.smt.ff.core`
  - `ff_ast.py`: typed AST nodes
  - `ff_poly.py`: sparse polynomial IR and partition extraction
  - `ff_algebra.py`: exact local algebraic lemmas and bounded partition solving
  - `ff_ir.py`: IR metadata/statistics helpers
  - `ff_reduction_scheduler.py`: reduction policy logic
  - `ff_modkernels.py`: modular reduction kernels
  - `ff_numbertheory.py`: primality and number-theory utilities
  - `ff_sympy.py`: symbolic helpers

Refinement stages in `FFPerfSolver`:

1. abstract integer solve;
2. exact GF(p) model validation;
3. failing-partition selection;
4. partition lemmas (affine first, bounded nonlinear second);
5. partition-scoped cut definition/activation;
6. stricter schedule fallback if needed.

The top-level `aria.smt.ff` package re-exports primary frontend and solver APIs.
