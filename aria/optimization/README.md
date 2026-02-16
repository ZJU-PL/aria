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
