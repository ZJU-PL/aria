# Prime Implicants and Prime Implicates

This module provides SAT-based algorithms for enumerating prime implicants and
prime implicates of Boolean formulas represented as PySAT `CNF` objects.

## Definitions

- A **prime implicant** is a minimal conjunction of literals that implies the
  formula.
- A **prime implicate** is a minimal clause that is implied by the formula.

Minimality matters in both cases:

- removing any literal from a prime implicant makes it stop implying the formula
- removing any literal from a prime implicate makes it stop being implied by the
  formula

## Why They Matter

Prime implicants and prime implicates are standard compact explanations of
Boolean behavior.

Common applications include:

- **Knowledge compilation and minimization**: derive canonical or near-canonical
  logical summaries of formulas.
- **Abductive reasoning and diagnosis**: prime implicants act as minimal
  sufficient explanations for an observation or target property.
- **Explanation in SAT/SMT-backed systems**: prime implicates provide minimal
  consequences, while prime implicants provide minimal witnesses.
- **Logic synthesis and circuit optimization**: prime implicants are central to
  two-level minimization and sum-of-products simplification.
- **Feature interaction and rule mining**: identify minimal combinations that
  force a behavior, and minimal clauses that always hold.
- **Verification and debugging**: isolate minimal sufficient conditions for
  success/failure and minimal invariants/consequences implied by a model.
- **Model-based reasoning**: build clause/term covers that support downstream
  enumeration, explanation, or decision procedures.

## API

```python
from pysat.formula import CNF

from aria.bool.prime import (
    enumerate_prime_implicants,
    enumerate_prime_implicates,
)

formula = CNF(from_clauses=[[1, 2], [-1, 3]])

prime_implicants = enumerate_prime_implicants(formula)
prime_implicates = enumerate_prime_implicates(formula)

print(prime_implicants)  # [[-1, 2], [1, 3], [2, 3]]
print(prime_implicates)  # [[-1, 3], [1, 2], [2, 3]]
```

Each result is a list of signed-integer literal lists:

- `x` means variable `x`
- `-x` means `not x`

## Implementation Notes

The implementation is SAT-based:

- prime implicants are obtained from satisfying assignments and minimized
  against the negated formula
- prime implicates are obtained from falsifying assignments of the negation and
  minimized against the original formula

This keeps the code simple and works well for small to medium formulas.

## Limits

- Enumeration is inherently exponential in the worst case.
- The current API expects formulas in CNF via PySAT's `CNF`.
- For very large instances, use this as an analysis/explanation tool rather than
  expecting full enumeration to scale.
