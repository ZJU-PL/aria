# SMT Module

ARIA ships several SMT subpackages aimed at different fragments and solving strategies:

- `arith`: incremental linearization for non-linear arithmetic, plus helpers for LRA theory, MiniSAT CNF export, and Mathematica bridges.
- `bv`: bit-vector infrastructure including mapped/unmapped BLAST bit-blasting and QF_BV / QF_AUFBV solver front-ends.
- `bwind`: bit-width independence bit-vector solving (by translating to quantified integer formulas)
- `ff`: finite-field SMT solvers, parsers, and SymPy-backed algebraic tooling.
- `fp`: floating-point procedures that reduce QF_FP / BV-FP formulas via Z3 tactics and PySAT.
- `lia_star`: LIA* solver port (can model BAPA constraints, namely, Boolean algebra with Presburger arithmetic; namely, sets with cardinality constraints)
- `mba`: Mixed Boolean-arithmetic (MBA) simplification
- `pcdclt`: parallel CDCL(T) solver stack with preprocessing, worker coordination, and tests.
- `portfolio`: QF_BV portfolio runner combining multiple Z3 tactics and SAT engines.
- `simplify`: formula simplification passes (context-aware, Dillig-style, DNF rewriting, dependency analysis).
- `unknown_resolver`: mutation-driven resolver for turning `unknown` solver answers into SAT/UNSAT outcomes.
