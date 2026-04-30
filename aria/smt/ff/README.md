# Finite-Field SMT in ARIA

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


This package uses a two-layer design:

1. A fuller SMT-LIB front-end that parses the QF_FF-style benchmark fragment
   used in this repository.
2. A set of strict solver backends that encode the resulting typed AST into
   Z3 integer or bit-vector formulas.

## Module Layout

The package is organized by responsibility:

- `aria.smt.ff.frontend`: parser and preprocess entry points
- `aria.smt.ff.solvers`: backend implementations and auto-selection
- `aria.smt.ff.core`: shared AST/IR/reduction helpers

For details, see [ARCHITECTURE.md](aria/smt/ff/ARCHITECTURE.md).
For the current partition-driven refinement design, see
[REFINEMENT.md](aria/smt/ff/REFINEMENT.md).
For a concise formalization of the current implementation, see
[FORMALIZATION.md](aria/smt/ff/FORMALIZATION.md).

## Front-End

The parser in [frontend/ff_parser.py](aria/smt/ff/frontend/ff_parser.py)
supports:

- multiple finite-field sorts in one formula
- `declare-fun` and `declare-const`
- non-recursive `define-fun` macro expansion
- `ff.add`, `ff.mul`, `ff.neg`, `ff.sub`, `ff.div`
- `ff.bitsum`, lowered to a weighted linear combination
- Boolean connectives `and`, `or`, `xor`, `not`, `=>`, `ite`
- negative finite-field literals such as `(as ff-1 F)`

Entry points:

- `parse_ff_file(...)`: full front-end, accepts mixed-field formulas
- `parse_ff_file_strict(...)`: same parser, but rejects formulas that mention
  more than one finite-field sort

`parse_ff_file_strict(...)` is useful when you want an explicit single-field
contract and want mixed-field inputs to fail early.

## Preprocessing

The normalization pass in
[frontend/ff_preprocess.py](aria/smt/ff/frontend/ff_preprocess.py)
is run before every backend.

It performs:

- flattening of associative field and Boolean operators
- constant folding for field arithmetic and Boolean connectives
- canonicalization of field equalities into `lhs - rhs = 0`
- exact simplification of small linear/idempotent patterns such as
  `c*x = 0`, `x + c = 0`, `x - y = 0`, and `x^2 = x`
- lowering of Booleanity constraints `x * (x - 1) = 0` into
  `(x = 0) or (x = 1)`
- detection of the standard `is_zero` witness gadget
  `m*x - 1 + z = 0` and `z*x = 0`, adding the sound implied facts
  `z in {0,1}` and `(z = 1) <=> (x = 0)`

These rewrites are intended to make the encoded formula more explicit without
changing satisfiability.

`preprocess_formula_with_metadata(...)` is available for experiment pipelines
that need normalization diagnostics (split counts, affine rewrites,
duplicate-removal counts, and partition statistics).

## Backends

Backends:

- [solvers/ff_bv_solver.py](aria/smt/ff/solvers/ff_bv_solver.py):
  faithful bit-vector encoding with modular reduction after every arithmetic
  step
- [solvers/ff_bv_solver2.py](aria/smt/ff/solvers/ff_bv_solver2.py):
  BV/Int bridge encoding using `BV2Int` and `Int2BV`
- [solvers/ff_int_solver.py](aria/smt/ff/solvers/ff_int_solver.py):
  direct integer encoding over bounded residues
- [solvers/ff_perf_solver.py](aria/smt/ff/solvers/ff_perf_solver.py):
  performance-oriented integer encoding with adaptive modulo scheduling,
  prime-structured reduction kernels, and partition-driven local CEGAR
  refinement over modulo-aware cut variables

The performance backend now uses a hybrid refinement ladder:

1. affine partition lemmas;
2. bounded nonlinear partition lemmas;
3. assertion-local exact lemmas;
4. partition-scoped cut definition/activation;
5. schedule fallback and stable integer fallback.

All backends:

- reset their solver state on every `check(...)`
- support more than one finite-field sort in a single formula
- use fast primality checking suitable for large benchmark moduli

`FieldDiv` is deliberately rejected by the backends unless the encoding is
extended with an explicit nonzero side condition. The previous silent
`a * b^(p-2)` behavior was unsound at `b = 0`.

## Automatic Backend Selection

[solvers/ff_solver.py](aria/smt/ff/solvers/ff_solver.py)
provides `FFAutoSolver`, which chooses a backend by the largest field bit-width:

- up to 31 bits: wide bit-vectors
- up to 160 bits: BV/Int bridge
- above 160 bits: performance backend (`FFPerfSolver`) by default, with
  fallback to stable integer encoding when needed

This is a practical large-prime strategy. It avoids forcing every large field
through the wide-BV `URem` path, which is the main source of the old large-prime
timeouts.

`FFAutoSolver` accepts:

- `enable_perf_backend`: enable or disable the perf backend route
- `perf_policy`: one of `auto`, `always`, `never`, `large-prime`

`FFPerfSolver` environment knobs:

- `ARIA_FF_SCHEDULE={eager,balanced,lazy,strict-recovery}`
- `ARIA_FF_KERNEL_MODE={auto,generic,structured}`
- `ARIA_FF_MAX_NONLINEAR_EQS`
- `ARIA_FF_MAX_NONLINEAR_VARS`
- `ARIA_FF_MAX_NONLINEAR_MODULUS`
- `ARIA_FF_MAX_NONLINEAR_SEARCH_SPACE`
- `ARIA_FF_MAX_NONLINEAR_WORK_BUDGET`
- `ARIA_FF_ROOTSET_BUDGET`

`FFPerfSolver` also exposes local-refinement controls:

- `cegar`: enable local SAT-model validation and cut refinement
- `max_refinement_rounds`: cap the number of refinement rounds per schedule
- `cut_seed_budget`: number of initially abstracted high-value subterms
- `cut_refine_budget`: number of mismatching cuts defined per refinement round
- `lemma_refine_budget`: number of learned lemmas per refinement round

For experiments, `FFPerfSolver` exposes:

- `stats()`: encoding/refinement counters, including partition-level metrics
- `trace()`: per-round JSON-friendly refinement trace

## API Quickstart

```python
from aria.smt.ff import FFAutoSolver, FFPerfSolver, parse_ff_file

formula = parse_ff_file("benchmarks/smtlib2/ff/simple.smt2")

# Automatic backend routing (perf backend enabled by default for large primes)
auto = FFAutoSolver()
print(auto.check(formula), auto.backend_name)

# Force the performance backend with explicit tuning knobs
perf = FFPerfSolver(schedule="balanced", kernel_mode="auto", recovery=True)
print(perf.check(formula))
print(perf.stats())
```

## Reproducible Performance Runs

Use the deterministic benchmark script:

```bash
python3 scripts/run_ff_perf_bench.py \
  --bench-dir benchmarks/smtlib2/ff \
  --backends bv,bv2,int,auto,perf \
  --timeouts 10,30,60 \
  --repetitions 3 \
  --out results/ff_perf_bench.json
```

Output JSON contains per-instance verdicts/timings and summary metrics (solved,
timeouts, PAR-2), plus averaged solver stats and trace lengths for each timeout
budget.

## Regression Driver

[ff_regress.py](aria/smt/ff/ff_regress.py)
now accepts `auto` in addition to `bv`, `bv2`, `int`, and `both`.
