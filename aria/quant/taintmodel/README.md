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
