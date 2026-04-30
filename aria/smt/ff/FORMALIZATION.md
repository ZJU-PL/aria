# Formalization of the `aria.smt.ff` Performance Backend

This note states the current algorithm precisely enough to support discussion,
ablation, and future proof work without overstating the implementation.

The current claim is:

`partition-driven finite-field CEGAR with exact local affine and bounded nonlinear refinement`

The procedure is sound for SAT and UNSAT answers, but incomplete.

## 1. Input Fragment

Let `F` be a quantifier-free formula over prime fields `GF(p)` with:

- field variables and constants;
- `+`, `-`, unary negation, multiplication, and non-negative powers;
- Boolean connectives over field equalities.

Division is rejected unless the encoding is extended with an explicit nonzero
side condition.

After preprocessing, the solver receives a finite list of assertions
`A = [a1, ..., an]` and field declarations for all variables.

## 2. Exact Semantics

For each field expression `e`, let `Eval_FF(e, nu)` be its exact value under an
assignment `nu`, computed modulo the field modulus of `e`.

A model `nu` satisfies `F` iff every assertion in `A` evaluates to true under
this exact finite-field semantics.

## 3. Polynomial Partitions

Whenever an assertion `ai` is a field equality `e1 = e2`, the solver lowers it
to a polynomial equation

`poly(ai) = poly(e1) - poly(e2) = 0 over GF(p)`.

Polynomial assertions over the same field are partitioned by shared variables.
Equivalently, for each modulus `p`, the solver builds a graph whose vertices
are polynomial equalities and whose edges connect equalities sharing at least
one variable. A partition is a connected component of this graph.

Each partition `P` therefore has:

- `I(P)`: assertion indices;
- `V(P)`: variables appearing in those assertions.

## 4. Abstraction State

For one schedule attempt, the abstraction state is

`S = (sigma, C, D, L, r)`

where:

- `sigma` is the current reduction schedule;
- `C` is the set of active cut expressions;
- `D` is the subset of defined cuts in `C`;
- `L` is the set of learned exact lemmas;
- `r` is the current refinement round.

Each cut for a field subterm `e` introduces an integer proxy `k_e` constrained
to the field range `[0, p-1]`.

Undefined cuts act as abstraction variables. Defined cuts recover the exact
meaning of their subterms.

## 5. Integer Abstraction

Given `S`, the solver builds an integer formula `Abs(S, F)` in QF_NIA:

1. every field variable `x : GF(p)` becomes an integer variable with
   `0 <= x < p`;
2. every active cut variable `k_e` gets the same range restriction;
3. field expressions are translated recursively to integers;
4. if a subexpression `e` is cut, the translation uses `k_e` unless `e` is
   currently being expanded to define that same cut;
5. if `e` is a defined cut, the abstraction adds
   `k_e = Red_sigma(Tr_sigma(e))`;
6. all original assertions and all learned lemmas are asserted.

The reduction schedule controls where modulo reductions are inserted during
translation.

## 6. Over-Approximation Invariant

For every state `S`, `Abs(S, F)` over-approximates `F`:

- every concrete finite-field model of `F` extends to an integer model of
  `Abs(S, F)`.

This holds because:

- field variables are embedded as bounded integers;
- undefined cuts only relax semantics;
- defined cuts impose exact equalities for those subterms;
- learned lemmas are exact consequences of the original formula.

This invariant is the basis of UNSAT soundness.

## 7. Validation

If `Abs(S, F)` is SAT, the solver obtains an integer model `mu` and checks the
original assertions under exact finite-field semantics.

Define:

- `Failed(mu) = { ai in A | Eval_FF(ai, mu) = false }`
- `Mismatch(mu) = { e in C \\ D | mu(k_e) != Eval_FF(e, mu) }`

If `Failed(mu)` is empty, the solver returns SAT.

Otherwise it refines using the failed assertions and mismatching cuts.

## 8. Partition Selection

Failed assertions are mapped to their polynomial partitions when possible.
Mismatching cuts are associated with partitions containing the corresponding
subexpression.

For a failing partition `P`, the implementation uses the score

`score(P) = (fa(P) + mm(P), hist(P), szA(P), szV(P), -ex(P))`

where:

- `fa(P)` is the number of failed assertions in `P`;
- `mm(P)` is the number of mismatching cuts associated with `P`;
- `hist(P)` is the prior failure count of `P`;
- `szA(P)` is the number of assertions in `P`;
- `szV(P)` is the number of variables in `P`;
- `ex(P)` is the current exactness level already spent on `P`.

The solver refines a partition with maximal lexicographic score. If no
partition is available, refinement falls back to the global failure set.

## 9. Exact Local Reasoning

Before defining more cut semantics, the solver tries to learn exact lemmas.

### 9.1 Affine partition lemmas

If a selected partition is affine and small enough, the solver performs modular
row reduction and may learn:

- `affine-root`
- `affine-relation`
- `affine-contradiction`

### 9.2 Bounded nonlinear partition lemmas

If a selected partition is within explicit bounds on equations, variables,
modulus, search space, and work budget, the solver enumerates all local models
of that partition and projects facts true in all of them. It may learn:

- `partition-root`
- `partition-rootset`
- `partition-relation`
- `partition-contradiction`

### 9.3 Single-assertion lemmas

The solver also derives exact one-assertion consequences such as:

- zero-product disjunctions;
- vanishing-power consequences;
- univariate linear roots;
- univariate monomial-zero consequences;
- constant contradictions.

Every learned lemma is deduplicated before being added to `L`.

## 10. Cut Refinement

If no new lemma is learned in a round, the solver refines cuts:

1. define mismatching active cuts from the selected partition, up to budget;
2. otherwise activate fresh cuts from that partition by structural priority;
3. otherwise fall back to the global cut pool.

Defining a cut decreases abstraction on that subterm. Activating a cut enlarges
the future refinement vocabulary.

## 11. Main Loop

For one schedule `sigma`, the inner loop is:

```text
for r = 0 .. max_refinement_rounds:
    solve Abs(S, F)
    if UNSAT: return UNSAT
    if UNKNOWN: return UNKNOWN

    validate the SAT model exactly in GF semantics
    if validation succeeds: return SAT

    choose one failing partition
    if exact local reasoning learns a new lemma:
        add it to L
        continue

    if cut refinement makes progress:
        update C and/or D
        continue

    break

return UNKNOWN
```

The outer driver escalates schedules through:

`lazy -> balanced -> strict-recovery -> eager`

and may finally fall back to the stable integer backend.

## 12. Soundness

### SAT soundness

If the solver returns SAT, the original finite-field formula is SAT, because
SAT is returned only after exact validation of the original assertions.

### UNSAT soundness

If the solver returns UNSAT, the original finite-field formula is UNSAT,
because every abstraction over-approximates the original formula.

### Lemma soundness

Every learned lemma is implied by its source assertions under exact
finite-field semantics:

- affine lemmas come from exact modular row reduction;
- bounded nonlinear lemmas come from complete local enumeration inside the
  configured bounds;
- single-assertion lemmas are exact consequences of the matched pattern.

## 13. Limits

The current implementation does not claim:

- completeness;
- general symbolic nonlinear elimination;
- Groebner-basis reasoning;
- strongest-possible projection for learned partition relations.

The honest summary is therefore:

`a sound but incomplete partition-driven CEGAR solver for finite fields`
