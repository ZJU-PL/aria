# Finite-Field SMT in ARIA

This package uses a two-layer design:

1. A fuller SMT-LIB front-end that parses the QF_FF-style benchmark fragment
   used in this repository.
2. A set of strict solver backends that encode the resulting typed AST into
   Z3 integer or bit-vector formulas.

## Front-End

The parser in [ff_parser.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_parser.py)
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
[ff_preprocess.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_preprocess.py)
is run before every backend.

It performs:

- flattening of associative field and Boolean operators
- constant folding for field arithmetic and Boolean connectives
- canonicalization of field equalities into `lhs - rhs = 0`
- lowering of Booleanity constraints `x * (x - 1) = 0` into
  `(x = 0) or (x = 1)`
- detection of the standard `is_zero` witness gadget
  `m*x - 1 + z = 0` and `z*x = 0`, adding the sound implied facts
  `z in {0,1}` and `(z = 1) <=> (x = 0)`

These rewrites are intended to make the encoded formula more explicit without
changing satisfiability.

`preprocess_formula_with_metadata(...)` is available for experiment pipelines
that need normalization diagnostics (split counts and derived gadget facts).

## Backends

Backends:

- [ff_bv_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_bv_solver.py):
  faithful bit-vector encoding with modular reduction after every arithmetic
  step
- [ff_bv_solver2.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_bv_solver2.py):
  BV/Int bridge encoding using `BV2Int` and `Int2BV`
- [ff_int_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_int_solver.py):
  direct integer encoding over bounded residues
- [ff_perf_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_perf_solver.py):
  performance-oriented integer encoding with adaptive modulo scheduling and
  prime-structured reduction kernels

All backends:

- reset their solver state on every `check(...)`
- support more than one finite-field sort in a single formula
- use fast primality checking suitable for large benchmark moduli

`FieldDiv` is deliberately rejected by the backends unless the encoding is
extended with an explicit nonzero side condition. The previous silent
`a * b^(p-2)` behavior was unsound at `b = 0`.

## Automatic Backend Selection

[ff_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_solver.py)
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
timeouts, PAR-2) for each timeout budget.

## Regression Driver

[ff_regress.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_regress.py)
now accepts `auto` in addition to `bv`, `bv2`, `int`, and `both`.
