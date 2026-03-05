# ARIA FF Architecture

The finite-field SMT package is split into three subpackages:

- `aria.smt.ff.frontend`
  - `ff_parser.py`: SMT-LIB finite-field parser
  - `ff_preprocess.py`: normalization and derived-fact rewrites
- `aria.smt.ff.solvers`
  - `ff_bv_solver.py`: bit-vector backend
  - `ff_bv_solver2.py`: BV/Int bridge backend
  - `ff_int_solver.py`: integer backend
  - `ff_perf_solver.py`: performance backend (adaptive scheduling + kernels)
  - `ff_solver.py`: automatic backend selector (`FFAutoSolver`)
- `aria.smt.ff.core`
  - `ff_ast.py`: typed AST nodes
  - `ff_ir.py`: IR metadata/statistics helpers
  - `ff_reduction_scheduler.py`: reduction policy logic
  - `ff_modkernels.py`: modular reduction kernels
  - `ff_numbertheory.py`: primality and number-theory utilities
  - `ff_sympy.py`: symbolic helpers

The top-level `aria.smt.ff` package re-exports primary frontend and solver APIs.
