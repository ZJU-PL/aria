# Playing with Quantifiers

The `aria.quant` package collects prototypes, ports, and utilities for solving or simplifying quantified SMT problems. Key components include:

- `chctools/`: Constrained Horn clause (CHC) tooling – command-line wrappers, SMT-LIB/Horn parsers, pretty-printers, and model validators that sit on top of Z3/Spacer.
- `efbool/`: Exists-forall solving over pure Booleans, with PySAT-backed exists/forall engines and sequential or parallel counterexample-guided refinement.
- `efbv/`: Exists-forall bit-vector stack offering sequential CEGIS loops, QBF/SAT encodings, and parallel sampling-based engines (`efbv_seq/`, `efbv_parallel/`).
- `eflira/`: CEGAR-style exists-forall solver for linear integer arithmetic (LIA) using paired Z3 solvers.
- `efsmt_parser.py` / `efsmt_solver.py` / `efsmt_utils.py`: Shared EFSMT front-end code—SMT-LIB parsing, Z3-based instantiation, and bridges to external binaries.
- `ematching/`: Trigger selection helpers that annotate quantified formulas with patterns suitable for Z3’s E-matching.
- `fossil/`: Port of the FOSSIL framework for synthesising inductive lemmas via natural proofs, SyGuS lemma synthesis, and benchmark suites.
- `polyhorn/`: Polynomial Horn clause solver (PolyHorn) with pysmt integrations for proving quantified real/integer constraints.
- `qe/`: Quantifier elimination experiments—Shannon expansion for CNF, LME-based projection (sequential and parallel), and adapters for QEPCAD, Mathematica, and Redlog.
- `ufbv/`: Parallel under/over-approximation framework for quantified bit-vectors (UFBV) coordinating multiple Z3 workers.

Most subpackages are self-contained research artifacts; expect varied maturity and external solver dependencies.
