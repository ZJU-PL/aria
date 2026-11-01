# Utility and Evaluation Scripts

This directory contains standalone benchmark runners, evaluation harnesses, and competition wrappers:

- `collect_queries/`: scripts for collecting SMT queries from dynamic/symbolic execution tools (KLEE, CBMC, angr)
- `efmc_eval/`: evaluation scripts for EFMC and EFSMT verifiers
- `smtcomp23/`: solver runners and wrappers from SMT-COMP 2023
- `smtcomp25/`: solver wrappers from SMT-COMP 2025
- `run_concolic_benchmarks.py`: manifest-driven seed-only versus native
  concolic comparison with coverage, path, failure, solver-query, cache, and
  timing metrics. See `benchmarks/concolic/README.md`.
- `run_ff_perf_bench.py`: benchmark harness for finite-field SMT backends
- `run_scsat.py`: shared-context batched satisfiability runner
- `run_solver.py`: utility script for invoking binary solvers
