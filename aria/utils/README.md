# Utilities

`aria.utils` contains a mix of shared helpers, solver adapters, Z3-specific
utilities, a few specialized top-level modules, and vendored code.

The package is organized around explicit sub-namespaces. New code should import
from those namespaces directly.

## Preferred Import Paths

### Shared Primitives
- `types.py`: solver- and platform-related enums
- `exceptions.py`: public exception types
- `sexpr.py`: S-expression parser

### Solver Utilities
- `solver/smtlib.py`: SMT-LIB process and portfolio helpers
- `solver/pysmt.py`: PySMT-backed solver helpers
- `solver/pysat.py`: PySAT-backed SAT helpers
- `solver/z3plus.py`: external-solver helpers around Z3 workflows

### Z3 Utilities
- `z3/expr.py`: Z3 expression inspection and manipulation helpers
- `z3/solver.py`: SAT/validity/entailment and DNF helpers
- `z3/opt.py`: Z3 optimization helpers
- `z3/bv.py`: bit-vector helpers and signedness inference
- `z3/uf.py`: uninterpreted-function helpers
- `z3/values.py`: BV/FP value conversion helpers
- `z3/cp.py`: constraint programming helpers (global constraints, variable domains)
- `z3/ext.py`: experimental expression utilities (quantifier manipulation, DNF)

### Parallel Execution
- `parallel/async_utils.py`: async bridge helpers
- `parallel/executor.py`: lightweight parallel execution
- `parallel/patterns.py`: curated parallel patterns API
- `parallel/master_slave.py`: master-slave pattern
- `parallel/producer_consumer.py`: producer-consumer pattern
- `parallel/fork_join.py`: fork-join pattern
- `parallel/pipeline.py`: pipeline pattern
- `parallel/actor.py`: actor model
- `parallel/dataflow.py`: dataflow graph
- `parallel/stream.py`: streaming primitives

## Top-Level Specialized Modules

These stay at the top level because they do not justify another package, but
they should still be treated as specialized/provisional:

- `misc.py`: catch-all helpers with weak ownership
- `sexpr2.py`: alternative S-expression helpers

## Vendored Code

- `pads/`: vendored graph-algorithms code; preserve upstream-style structure
- `ply/`: vendored parsing helpers used by SRK
- `pycparser/`: vendored C parser used by EFMC's C frontend

## Specialized Packages

- `bdd/`: binary decision diagram helpers

## Import Policy

- Use `aria.utils.solver.*` for solver-facing helpers
- Use `aria.utils.z3.*` for Z3-specific helpers
- Use top-level modules directly for the few specialized helpers above
- Avoid adding new unrelated helpers to top-level `aria.utils`
