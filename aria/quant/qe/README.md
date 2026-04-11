# Quantifier Elimination Experiments

`aria.quant.qe` collects several QE implementations with different input formats,
dependencies, and maturity levels. This directory is intentionally mixed. Some
paths are small in-process Python helpers, some wrap Z3 tactics, and some shell
out to external tools.

The code here does not expose one unified QE abstraction. Pick the entry point
that matches your formula representation and the kind of result you need.

## Capability matrix

| Entry point | Input shape | Theory or domain expectations | Partial projection support | Return shape | External dependencies | Maturity and notes |
| --- | --- | --- | --- | --- | --- | --- |
| `qe_expansion.QuantifierElimination` | PySAT `CNF` plus variable names mapped through `get_var_id()` | Boolean CNF only, existential elimination only | No. Eliminates the listed quantified Boolean variables only | PySAT `CNF` | `python-sat`, `Glucose3` for `is_satisfiable()` | Small in-process prototype based on Shannon expansion and clause resolution. Best for Boolean CNF workflows already using PySAT objects. |
| `qe_cooper.qelim_exists_lia_cooper` | Z3 formula `phi`, one Z3 variable or iterable `qvars`, optional `keep_vars` | Quantifier-free Presburger-style LIA over `Int` variables only. Accepts affine arithmetic atoms with `<=`, `<`, `>=`, `>`, `==` and Boolean structure normalized through NNF/DNF. Rejects mixed sorts, nonlinear arithmetic, explicit input `mod/div`/casts, nested quantifiers, UF/arrays, and `!=` | Yes. `keep_vars` matches `qe_lme`: when provided, every free variable not explicitly kept is projected away together with `qvars` | Z3 `BoolRef` | Z3 Python bindings | Direct Z3-facing Cooper-style prototype for integer arithmetic. Uses equality pivots plus coefficient normalization to a unit-coefficient core with congruence/divisibility guards. Designed for the supported fragment only; unsupported input fails fast with `ValueError`. |
| `qe_fm.qelim_exists_lra_fm` | Z3 formula `phi`, one Z3 variable or iterable `qvars`, optional `keep_vars` | Quantifier-free linear real arithmetic only: `Real` variables, affine arithmetic atoms, and Boolean structure handled by NNF/DNF cube expansion | Yes. Matches `qe_lme` exactly: `keep_vars` keeps only the requested free variables, `keep_vars=[]` projects away all non-quantified free variables, and `qvars` must stay disjoint from `keep_vars` | Z3 `BoolRef` | Z3 Python bindings | Direct in-process Fourier-Motzkin path for supported LRA fragments. Eliminates one variable at a time, preserves cubes that do not mention the eliminated variable, and fails fast on mixed sorts, nonlinear terms, nested quantifiers, or guarded DNF blowups. |
| `qe_lme.qelim_exists_lme` | Z3 formula `phi`, one Z3 variable or iterable `qvars`, optional `keep_vars` | Uses Z3 `qe2` on minterms. Tested in this repo on linear integer and real arithmetic style formulas plus mixed Boolean structure over Z3 atoms | Yes. `keep_vars` keeps only the requested free variables. When `keep_vars` is provided, every other free variable in `phi` is projected away together with `qvars`. `keep_vars=[]` projects away all non-quantified free variables. `qvars` and `keep_vars` must be disjoint | Z3 `BoolRef` | Z3 Python bindings | Primary in-process QE helper in this directory. This is the only entry point here with the implemented `keep_vars` partial projection API. |
| `qe_lme_parallel.qelim_exists_lme_parallel` | Z3 expression or SMT-LIB-like string `phi`, list of Z3 variables `qvars` | Model extraction and projection are done by spawning Z3 subprocesses and exchanging SMT-LIB or JSON data | No `keep_vars` API | String projection, or the literal string `"false"` on failure or empty result | Z3 executable resolvable through `aria.global_params.global_config` | Experimental. Recent code hardens solver-path resolution, temp-file cleanup, duplicate projection removal, timeout handling, and worker-failure tolerance, but the interface is still string-returning and not aligned with the sequential Z3 `BoolRef` API. |
| `external_qe.ExternalQESolver` | Quantified formula as a string, optional backend selection and backend-specific kwargs | Depends on backend and input syntax conversion. Auto-selection prefers QEPCAD for real domains, then QEPCAD, Redlog, Mathematica by availability | No explicit partial projection API. Projection is whatever the backend computes from the quantified input string | `(success: bool, result_formula: str)` | External binaries for QEPCAD, Mathematica, or Redlog | Adapter layer for external solvers. Good when you already have solver-specific text formulas and want backend selection or fallback logic. |

## Choosing an entry point

- Use `qe_expansion.QuantifierElimination` for Boolean CNF formulas built with
  PySAT.
- Use `qe_cooper.qelim_exists_lia_cooper` for direct Z3-facing existential QE on
  the supported quantifier-free integer linear arithmetic fragment when you want
  explicit fail-fast boundaries and exact `keep_vars` behavior matching
  `qelim_exists_lme`.
- Use `qe_fm.qelim_exists_lra_fm` for direct Z3-facing Fourier-Motzkin
  elimination on quantifier-free affine formulas over `Real` variables when you
  want a pure in-process LRA projection and can stay inside the supported
  fragment.
- Use `qe_lme.qelim_exists_lme` for in-process Z3 formulas when you want a Z3
  result and, if needed, partial projection through `keep_vars`.
- Use `qe_lme_parallel.qelim_exists_lme_parallel` only when you explicitly want
  the parallel experimental path and can handle string results.
- Use `external_qe.ExternalQESolver` when the source formula is already text and
  you have one of the external QE backends installed.

## Sequential LME partial projection

`qelim_exists_lme()` now supports partial projection through `keep_vars`.

- `keep_vars=None` keeps the historical behavior. Only `qvars` are eliminated.
- `keep_vars=[...]` keeps exactly those free variables and projects away the
  rest.
- `keep_vars=[]` requests a full projection to a closed formula.
- Passing the same variable in both `qvars` and `keep_vars` raises
  `ValueError`.

Small example:

```python
import z3

from aria.quant.qe.qe_lme import qelim_exists_lme

x, y, z = z3.Ints("x y z")
phi = z3.And(x == y + z, y > 0, z >= 0)

projected = qelim_exists_lme(phi, [x], keep_vars=[y])
print(projected)
```

This keeps `y` in the result and projects away both `x` and `z`.

## Cooper-style LIA elimination

`qelim_exists_lia_cooper()` is a stricter in-process path for quantifier-free
integer linear arithmetic.

- Input must stay in the supported `Int`-only affine fragment.
- The result may contain integer congruence/divisibility constraints, e.g.
  modulo conditions that characterize solvability.
- `keep_vars` follows the same projection rule as `qelim_exists_lme()`.

Small example:

```python
import z3

from aria.quant.qe.qe_cooper import qelim_exists_lia_cooper

x, y = z3.Ints("x y")
phi = 3 * x + 1 == y

projected = qelim_exists_lia_cooper(phi, [x])
print(projected)
```

This projects away `x` and keeps the exact integer solvability condition on
`y`.

## Sequential Fourier-Motzkin LRA projection

`qelim_exists_lra_fm()` provides a narrow direct Z3-facing path for linear real
arithmetic.

- Supported atoms are affine `<=`, `<`, `>=`, `>`, `==`, and `!=` over
  `Real` variables only.
- Boolean structure is normalized through NNF/DNF cube expansion, with `!=`
  branched locally into `<` or `>`.
- `keep_vars` follows the same semantics as `qelim_exists_lme()` because both
  entry points share the same projection-variable helper.
- The implementation intentionally fails fast on mixed Int/Real formulas,
  nonlinear arithmetic, nested quantifiers, and cube expansions that exceed the
  guarded bound.

Small example:

```python
import z3

from aria.quant.qe.qe_fm import qelim_exists_lra_fm

x, y, z = z3.Reals("x y z")
phi = z3.And(x >= y + 1, x <= z - 2)

projected = qelim_exists_lra_fm(phi, [x])
print(projected)
```

## Notes on the parallel LME path

`qe_lme_parallel` is still marked experimental in practice.

- It resolves the Z3 executable through global solver configuration, not the Z3
  Python API.
- It returns projection strings, not Z3 expressions.
- Failure cases collapse to the string `"false"`.
- The current hardening is operational, not architectural. The code now checks
  solver-path availability, validates basic bounds and timeouts, cleans up temp
  SMT files, deduplicates projections, and keeps surviving results when one
  worker fails.

## Notes on external backends

`ExternalQESolver` detects available backends at construction time and exposes
them through `get_available_backends()`. The main entry point,
`eliminate_quantifiers()`, returns `(success, result_formula)` and accepts
backend-specific options such as `domain`, `logic`, and `timeout`.

The adapters are thin wrappers. They do not normalize formula syntax across all
backends beyond the small conversions implemented in `external_qe.py`, so the
caller is responsible for providing backend-appropriate input strings.
