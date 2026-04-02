`aria.quant.taintmodel` is a prototype implementation of taint-based
SIC/WIC inference inspired by:

- CAV 2018: Model Generation for Quantified Formulas: A Taint-Based Approach
  https://arxiv.org/pdf/1802.05616

Current solver scope is intentionally restricted to the prenex
`exists X . forall Y . P(X, Y)` fragment, where `P` is quantifier-free.

- Free variables are treated as existential parameters.
- Nested quantifiers inside `P` are not supported.
- Alternating or non-prenex quantified formulas outside this fragment return
  `unknown` rather than using unsafe eliminations.

The solver uses taint-generated sufficient independence conditions (SICs) to
reduce the universal block to a quantifier-free obligation, and only reports
`unsat` when additional checks justify the reduction as complete for the case at
hand.

## Current Algorithm

The default solver is no longer a pure one-shot reduction. It now runs a small
counterexample-guided refinement loop around the taint engine:

1. Infer an initial SIC `psi(X)` for the matrix `P(X, Y)` and collect
   target-free guard candidates that arise during taint propagation.
2. Check **soundness** by asking whether
   `P(X, Y) /\ psi(X) /\ not P(X, Y')`
   is satisfiable for a fresh copy `Y'` of the universal block.
3. If a bad existential assignment is found, strengthen `psi`:
   - first try to relearn a guard from the accumulated positive/negative
     samples using the taint-generated candidate guards;
   - otherwise conjoin a blocker that excludes the bad sample.
4. Solve the reduced quantifier-free obligation `P(X, Y0) /\ psi(X)`.
5. If that reduced problem is unsat, check **completeness** by asking whether
   there exists an existential assignment satisfying
   `forall Y. P(X, Y) /\ not psi(X)`.
6. If such a good assignment exists, weaken `psi`:
   - first try to synthesize a DNF-like guard from the sample set;
   - otherwise disjoin a region that includes the good sample.
7. Repeat until a sound SAT witness is found, unsat is certified, or the
   refinement budget is exhausted.

The default constructor enables this loop via `QuantSolver(refine_sic=True)`.
Passing `refine_sic=False` restores the earlier one-shot behavior.

## Candidate Guards

The taint engine now records more than the final flattened SIC. During the
recursive pass it keeps target-free Boolean guards from:

- operator-specific theory rules (`Psi_f` candidates),
- combined local SICs after recursive composition, and
- the final inferred SIC.

These guards form the feature set for the refinement loop. The learner is
deliberately lightweight: it prefers short guards, synthesizes cubes that cover
known good samples while excluding known bad ones, and falls back to exact
sample blockers/regions when no useful generalization is available.

## Guarantees

- `sat` is returned only after the current SIC passes the explicit soundness
  check.
- `unsat` is returned only when the reduced problem is unsat and the
  completeness check shows that no satisfying existential assignment lies
  outside the current SIC.
- If the refinement loop cannot validate progress, the solver returns
  `unknown` rather than taking an unsafe shortcut.
