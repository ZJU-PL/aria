# Optimization

Optimization and Maximum Satisfiability (MaxSAT) solvers.

## Components

### MaxSAT Solvers
- `maxsmt/base.py`: Base MaxSAT classes
- `maxsmt/core_guided.py`: Core-guided MaxSAT algorithms
- `maxsmt/local_search.py`: Local search MaxSAT
- `maxsmt/z3_optimize.py`: Z3 optimization interface
- `maxsmt/ihs.py`: Instance-based heuristic search

### OMT (Optimization Modulo Theories)
- `omt_solver.py`: Main OMT solver
- `omtarith/`: OMT for arithmetic theories
  - `arith_opt_lp.py`: LP-based optimization
  - `arith_opt_ls.py`: Local search optimization
  - `arith_opt_qsmt.py`: QSMT-based optimization
- `omtbv/`: OMT for bit-vectors
  - `bv_opt_maxsat.py`: MaxSAT-based BV optimization
  - `bv_opt_qsmt.py`: QSMT-based BV optimization
  - `bit_blast_omt_solver.py`: Bit-blasting approach
  - `boxed/`: Boxed optimization variants
- `omtfp/`: OMT for floating-point theories
  - `fp_omt_parser.py`: SMT-LIB parser for FP objectives
  - `fp_opt_iterative_search.py`: IEEE-754 total-order iterative optimization
  - `fp_opt_qsmt.py`: Exact quantified-SMT floating-point optimization

### Floating-Point OMT Semantics
- FP optimization uses IEEE-754 `totalOrder` over exact floating-point encodings.
- This is intentionally different from `fp.lt` / `fp.leq`, which define only a partial
  numeric order and do not provide a total ordering over NaNs or distinguish all
  bit-level cases needed for general optimization.
- As a result, optimization over QF_FP objectives is well-defined for signed zeros,
  infinities, and NaNs, including distinct NaN payload/sign encodings.

### Floating-Point Pareto Semantics
- FP Pareto optimization compares objective tuples componentwise using the same
  IEEE-754 `totalOrder` semantics.
- A point is Pareto-optimal if no other feasible point is at least as good in every
  objective and strictly better in at least one objective under the per-objective
  direction (`maximize` or `minimize`).
- The reported frontier therefore preserves floating-point distinctions such as
  `-0.0` versus `+0.0` and different NaN encodings whenever they affect dominance.
- Pareto results are rendered as lists of objective tuples, and each FP value is
  shown with both a readable form and its exact IEEE bit pattern.

### MSA (Minimal Satisfying Assignment)
- `msa/mistral_msa.py`: Mistral MSA solver
- `msa/mistral_pysmt.py`: PySMT integration

### Utilities
- `omt_parser.py`: OMT problem parser
- `pysmt_utils.py`: PySMT utilities
- `bin_solver.py`: Binary solver wrapper

## Usage

```python
from aria.optimization import MaxSATSolver

solver = MaxSATSolver()
result = solver.solve(weighted_cnf)
```
