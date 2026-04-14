# SMT Simplification

This package collects formula simplification utilities and research prototypes
 for SMT-style expressions.

## Main Modules

- `ctx_simplify.py`: contextual simplification over a Z3 term using a solver
  context and model. It looks for equivalent subterms with the same model value
  and replaces larger terms with simpler ones when justified by the context.
- `dillig_simplify.py`: an implementation attempt of the online constraint
  simplification algorithm from Dillig, Dillig, and Aiken (SAS 2010).
- `dnf.py`: BDD-based conversion utilities for producing DNF formulas with
  PySMT, plus predicate-rewriting walkers used around that conversion.
- `mondep.py`: an implementation of monadic decomposition inspired by the
  CAV'14 algorithm.

## Scope

The package is intentionally mixed:

- small utility code that can be reused by other SMT modules;
- algorithm prototypes that are useful for experimentation and analysis;
- transformations that rely on Z3 or PySMT internals.

## Typical Usage

Contextual simplification:

```python
import z3

from aria.smt.simplify.ctx_simplify import ctx_simplify

x, y = z3.Ints("x y")
solver = z3.Solver()
solver.add(x == y + 1)
solver.check()
model = solver.model()

term = x - y
print(ctx_simplify(solver, model, term))
```

## Notes

- `dillig_simplify.py` is explicitly marked as likely buggy in the source and
  should be treated as experimental.
- `dnf.py` requires PySMT and a BDD backend such as `repycudd`.
- Some algorithms here are research-oriented and may prioritize idea
  preservation over a stable production interface.
