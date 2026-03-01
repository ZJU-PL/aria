# Finite-Field SMT in ARIA

This package now has a two-layer design:

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

Two entry points are exposed:

- `parse_ff_file(...)`: full front-end, accepts mixed-field formulas
- `parse_ff_file_strict(...)`: same parser, but rejects formulas that mention
  more than one finite-field sort

The strict entry point is useful when you want the old single-field discipline
as an explicit contract instead of an accidental parser limitation.

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

## Backends

Three direct backends are available:

- [ff_bv_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_bv_solver.py):
  faithful bit-vector encoding with modular reduction after every arithmetic
  step
- [ff_bv_solver2.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_bv_solver2.py):
  BV/Int bridge encoding using `BV2Int` and `Int2BV`
- [ff_int_solver.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_int_solver.py):
  direct integer encoding over bounded residues

All three backends now:

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
- above 160 bits: integer encoding

This is a practical large-prime strategy. It avoids forcing every large field
through the wide-BV `URem` path, which is the main source of the old large-prime
timeouts.

## Regression Driver

[ff_regress.py](/Users/rainoftime/Work/logic/aria/aria/smt/ff/ff_regress.py)
now accepts `auto` in addition to `bv`, `bv2`, `int`, and `both`.
