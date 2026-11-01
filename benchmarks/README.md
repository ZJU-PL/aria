# Benchmarks Directory

This directory contains benchmark suites and test corpora for ARIA's solvers and verification engines.

## Directories

### efmc

Program-verification benchmarks used by `aria/efmc`:

- `efmc/Boogie/` - Boogie verification programs
- `efmc/BV/` - Bit-vector verification benchmarks
- `efmc/C/` - C verification programs (see `efmc/C/README.md`)
- `efmc/chc/` - Constrained Horn Clause benchmarks
- `efmc/INT/` - Integer verification benchmarks
- `efmc/KSafety/` - Relational and k-safety benchmarks
- `efmc/sygus-inv/` - SyGuS invariant-synthesis benchmarks

### abduction
SMT-LIB2 abduction benchmark instances.

### concolic
Concolic execution benchmark manifests and targets (see `concolic/README.md`).

### dimacs
Boolean SAT benchmark instances in DIMACS CNF format.

### fzn
MiniZinc FlatZinc constraint optimization benchmarks.

### FP
Floating-point loop invariant synthesis benchmarks (including Pine benchmarks).

### Polyhorn
Polynomial Horn and termination benchmarks.

### qbf
Quantified Boolean formula benchmarks in QDIMACS format.

### smtlib2
Standard SMT-LIB2 benchmark queries (EPR, finite fields, interpolation, optimization).

### sygus-pbe
Syntax-guided synthesis programming-by-example (PBE) benchmarks for strings and bit-vectors.
