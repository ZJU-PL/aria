# Partition-Driven Refinement in `aria.smt.ff`

This note documents the current refinement architecture implemented by
[`solvers/ff_perf_solver.py`](aria/smt/ff/solvers/ff_perf_solver.py) and the
local algebra support in [`core/ff_algebra.py`](aria/smt/ff/core/ff_algebra.py).

For a concise formalization of the same implementation, see
[`FORMALIZATION.md`](aria/smt/ff/FORMALIZATION.md).

It is written for two audiences:

- developers extending the current solver;
- researchers trying to understand what claim the implementation currently
  supports.

## Current Story

The performance backend is no longer just "delay modulo, then define more
cuts." The current loop is:

1. preprocess the formula and detect polynomial structure;
2. encode the formula into integer arithmetic with schedule-aware modulo
   reduction and selected cut variables;
3. solve the abstraction;
4. validate SAT candidates under exact GF(p) semantics;
5. localize the failure to one polynomial partition when possible;
6. try exact local algebraic lemmas for that partition;
7. only if that fails, define or activate cuts scoped to that partition;
8. repeat until SAT/UNSAT/`unknown`, then fall back to stricter schedules or
   the stable integer backend.

The key change is that refinement is now *partition-driven* rather than purely
expression-driven.

## What Is a Partition?

A partition is a connected component of polynomial equalities that share
variables. Partitioning is computed by
[`core/ff_poly.py`](aria/smt/ff/core/ff_poly.py).

This serves two purposes:

- local algebra is applied only where there is real shared structure;
- refinement traces and stats can say *which* part of the problem became more
  exact.

## Refinement Order

For the currently selected failing partition, the solver refines in this order:

1. affine partition lemmas;
2. bounded nonlinear partition lemmas;
3. single-assertion exact lemmas;
4. partition-scoped cut definition;
5. partition-scoped cut activation;
6. global cut activation fallback.

That order is deliberate. The implementation prefers exact projected
consequences over raw cut materialization.

## Local Algebra

The local algebra layer currently supports:

- simple exact one-assertion lemmas:
  - zero-product disjunctions;
  - vanishing powers;
  - univariate linear roots;
  - univariate monomial zeros;
- affine elimination on small partitions via modular row reduction;
- bounded nonlinear search on very small partitions and moduli.

The nonlinear path is conservative and production-oriented:

- it is guarded by bounds on partition size, modulus size, total search space,
  and an explicit work budget;
- it only learns consequences that are exact across *all* local models:
  - `partition-root`
  - `partition-rootset`
  - `partition-relation`
  - `partition-contradiction`

This makes it safe to use in the solver loop without relying on a heavyweight
symbolic backend.

## Stats and Trace

`FFPerfSolver.stats()` exposes several groups of counters:

- encoding:
  - `reductions_*`
  - `kernel_*`
- abstraction/refinement:
  - `cuts_*`
  - `lemma_*`
  - `partition_*`
  - `selected_partition_*`
- local algebra:
  - `partition_cache_*`
  - `partition_solver_*`
- usefulness:
  - `useful_lemmas`
  - `cuts_avoided_by_lemmas`
  - `lemma_rounds_led_to_unsat`
  - `lemma_rounds_led_to_sat`

`FFPerfSolver.trace()` returns a per-round JSON-friendly trace with:

- schedule;
- round number;
- failing assertion count;
- failing partition count;
- selected partition size/variable count/modulus;
- learned lemma count;
- defined/activated cut counts;
- optional terminal result.

The benchmark runner in `scripts/run_ff_perf_bench.py` persists these metrics
into its JSON output.

## Environment Controls

The backend currently reads:

- `ARIA_FF_SCHEDULE`
- `ARIA_FF_KERNEL_MODE`
- `ARIA_FF_MAX_NONLINEAR_EQS`
- `ARIA_FF_MAX_NONLINEAR_VARS`
- `ARIA_FF_MAX_NONLINEAR_MODULUS`
- `ARIA_FF_MAX_NONLINEAR_SEARCH_SPACE`
- `ARIA_FF_MAX_NONLINEAR_WORK_BUDGET`
- `ARIA_FF_ROOTSET_BUDGET`

These are intended for controlled experiments and ablations.

## Limits

What this implementation does **not** yet claim:

- a complete symbolic nonlinear partition solver;
- Gröbner-basis reasoning;
- completeness of the current abstraction domain;
- a proof that every learned partition relation is the strongest possible
  projection.

So the current best research framing is:

"partition-driven finite-field CEGAR with exact local affine and bounded
nonlinear refinement."

That is stronger than a generic encoding heuristic, but weaker than a fully
general algebraic decision procedure.

## Recommended Next Research Steps

If the goal is a full paper, the next steps are:

1. formalize the refinement domain and monotone progress property;
2. add ablation-ready experiment modes for each refinement tier;
3. compare against external finite-field SMT baselines;
4. replace some bounded enumeration cases with stronger symbolic nonlinear
   elimination.
