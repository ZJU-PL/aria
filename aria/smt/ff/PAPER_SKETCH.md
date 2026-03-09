# Paper Sketch for `aria.smt.ff`

This note turns the current implementation into a candidate research-paper
story. It is intentionally narrow: it describes what the code already supports
or is close to supporting, instead of inventing a different solver.

Related implementation notes:

- [REFINEMENT.md](aria/smt/ff/REFINEMENT.md)
- [ARCHITECTURE.md](aria/smt/ff/ARCHITECTURE.md)

## Candidate Title

Partition-Driven Abstraction Refinement for SMT over Finite Fields

## One-Sentence Claim

We present a finite-field SMT solver that combines integer translation with
partition-driven abstraction refinement, where spurious abstract models are
eliminated by exact local algebraic consequences derived from connected
polynomial subproblems.

## What Is Actually New Here

The strongest current claim is not:

"we improved an integer encoding."

It is:

"we refine finite-field SMT abstractions at the level of polynomial
partitions, preferring exact projected local algebraic consequences over
global cut materialization."

That claim is grounded in the current code:

- sparse polynomial IR and partition extraction:
  [`core/ff_poly.py`](aria/smt/ff/core/ff_poly.py)
- exact local algebra:
  [`core/ff_algebra.py`](aria/smt/ff/core/ff_algebra.py)
- partition-driven refinement loop:
  [`solvers/ff_perf_solver.py`](aria/smt/ff/solvers/ff_perf_solver.py)

## Algorithm Summary

### Input

A quantifier-free finite-field formula over prime fields.

### Output

`sat`, `unsat`, or `unknown`, with optional model/stats/trace.

### High-Level Algorithm

```text
Algorithm PartitionFF-CEGAR(phi):
  phi0 := preprocess(phi)
  P := polynomial partitions of phi0
  A := integer abstraction of phi0 with selected cuts

  repeat:
    res, M := solve(A)
    if res != sat:
      return res or fallback(res)

    if M is exact GF(p)-model of phi0:
      return sat

    F := failed assertions under exact validation
    p := choose failing partition from F

    L := exact local lemmas for p
         (affine elimination, bounded nonlinear solving, local algebraic rules)
    if L is nonempty:
      A := A ∧ L
      continue

    C := partition-scoped cut refinement for p
    if C changes abstraction:
      A := refine(A, C)
      continue

    return fallback(unknown)
```

### Refinement Order

For one selected partition, the implementation refines in this order:

1. affine partition lemmas;
2. bounded nonlinear partition lemmas;
3. single-assertion exact lemmas;
4. partition-scoped cut definition;
5. partition-scoped cut activation;
6. global fallback.

This ordering matters. It is the core nontrivial design choice in the current
system.

## Abstraction Domain

The current abstraction domain consists of:

- integer-translation constraints with schedule-controlled modular reductions;
- bounded cut variables for selected field expressions;
- learned exact lemmas over assertions and partitions.

The current implementation is not yet a full symbolic algebraic decision
procedure. It is a layered abstraction domain with exact local repairs.

## Local Solvers

The partition-local reasoning stack is:

1. affine elimination on small partitions via modular RREF;
2. bounded nonlinear model enumeration on very small partitions/moduli;
3. projection of exact consequences back to the global abstraction:
   - root equalities;
   - small root sets;
   - affine relations;
   - contradictions.

The nonlinear local solver is deliberately bounded by:

- equation count;
- variable count;
- modulus bound;
- search-space bound;
- explicit work budget.

That boundedness is not a weakness for the paper if you state it clearly. It
just means the paper is about local exact refinement under explicit resource
guards, not about complete nonlinear elimination.

## Theorem Candidates

The current implementation is closest to supporting the following statements.

### 1. Soundness of Learned Lemmas

Every learned lemma added by the local algebra layer is valid in the original
finite-field semantics of the selected partition.

Proof sketch:

- affine lemmas are derived by exact elimination over GF(p);
- nonlinear lemmas are derived from complete enumeration of the local model set
  inside the configured bounded search regime;
- one-assertion lemmas are direct finite-field identities.

This is the easiest theorem and should be included.

### 2. Monotone Refinement Progress

Each successful refinement round strictly strengthens the current abstraction
for the selected partition.

Reason:

- learning a new lemma excludes at least the current spurious model;
- defining a previously abstract cut strictly increases exactness;
- activating a new cut increases the represented semantics of the selected
  partition.

This is the next-best theorem candidate.

### 3. Conditional Finite Convergence

Under bounded partitions and finite cut pool, if every selected failing
partition is eventually fully materialized, the refinement loop terminates.

This is weaker than a full completeness theorem, but still useful if stated
carefully.

## Evaluation Story

The current solver already exposes the instrumentation needed for a useful
evaluation.

### Primary Metrics

- solved instances
- runtime
- PAR-2
- refinement rounds
- learned lemmas
- cuts defined / cuts activated
- useful lemmas
- cuts avoided by lemmas
- selected partition size / variable count
- partition solver hits / cache hits / search nodes

### Ablations

The paper should include:

1. full solver
2. no partition lemmas
3. no nonlinear partition solver
4. no affine partition solver
5. cut-only refinement
6. no preprocessing extras
7. no modulo schedule adaptation

### External Baselines

At minimum:

- cvc5 finite-field support
- Yices2 finite-field support

Internal backends (`bv`, `bv2`, `int`) are useful ablations, not primary
baselines.

## What to Emphasize in Writing

Emphasize:

- refinement granularity is the partition, not the individual cut;
- exact local algebra is used to strengthen the abstraction before semantics
  materialization;
- the solver is measurable: stats and traces expose why it works.

Do not overclaim:

- no full Gröbner-basis engine;
- no general completeness for nonlinear finite-field reasoning;
- bounded nonlinear search is a local exact subroutine, not a universal solver.

## Suggested Paper Sections

1. Introduction
2. Background on QF_FF and encoding-based solving
3. Partition-Driven Refinement Framework
4. Exact Local Algebraic Refinement
5. Implementation in ARIA
6. Experimental Evaluation
7. Limitations and Future Work

## Suggested Figures

1. Pipeline figure:
   parser -> preprocess -> partitioning -> abstraction -> validation ->
   partition refinement -> fallback

2. Refinement lattice:
   local lemmas before cuts before fallback

3. Partition heatmap:
   partition size vs refinement success

4. Ablation plot:
   solved instances vs enabled refinement tiers

## Suggested Tables

1. Solver configuration table
2. Ablation summary
3. Lemma usefulness summary
4. Partition solver usage summary

## What Would Make This Full-Paper Ready

The implementation is close, but a full paper still needs:

1. one formal theorem written and proved cleanly;
2. serious external evaluation;
3. ablations isolating each refinement tier;
4. failure analysis using the current `trace()` and `stats()` outputs.

Until then, the most honest positioning is:

"paper-ready prototype with a strong workshop/tool-paper story and a plausible
full-paper algorithmic core."
