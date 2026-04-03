# Benchmarks Directory

This directory contains benchmark files and test cases for the ARIA project.
The benchmark files are primarily in SMT2 and SyGuS(Inv) format.



## efmc

Program-verification benchmarks used by `aria/efmc` are grouped under
`efmc/`:

- `efmc/Boogie/` - Boogie verification programs
- `efmc/BV/` - Bit-vector verification benchmarks
- `efmc/C/` - C verification programs
- `efmc/chc/` - Constrained Horn Clause benchmarks
- `efmc/INT/` - Integer verification benchmarks
- `efmc/KSafety/` - Relational and k-safety benchmarks
- `efmc/sygus-inv/` - SyGuS invariant-synthesis benchmarks

For more details on the C benchmarks, refer to `efmc/C/README.md`.

## Demo
Contains demonstration Constraint Horn Clauses (CHC) files (in SMT-LIB2 format) showcasing basic examples:
- `fib13_8u.smt2`, `fib10_8u.sm2`, `fib04_32u.smt2` - Fibonacci sequence examples with different bit-vector sizes

## FP

Related work


- FP/pine-benchmarks: SAS 20: Counterexample- and Simulation-Guided
Floating-Point Loop Invariant Synthesis.
https://github.com/izycheva/pine
- (TBD) Invariant Generation for Floating-Point Programs via Constraint Solving
Xuran Cai, Liqian Chen, Hongfei Fu. https://arxiv.org/pdf/2507.15017v1

## Polyhorn

Solving "Exists-Forall" Problems over Non-linear Reals? (with applications in
invariant infernce, ranking function synthesis, and non-linear loop synthesis, etc.)

It seems that polyhorn has been upgrated/renamed to "polyqent"
- ATVA 2025:  PolyQEnt: A Polynomial Quantified Entailment Solver. https://github.com/ChatterjeeGroup-ISTA/polyqent
- https://arxiv.org/pdf/2408.03796

Baseline tools
- Mathmematica
- redlog
- ...?


## Separation Logic

TBD

- https://github.com/sl-comp/sl-comp.github.io

## More Benchmarks

- https://github.com/sosy-lab/sv-benchmarks/tree/master/clauses

We can find much longer CHC instances (e.g., clauses/ALIA/sdv, where is from Microsfot's device driver verification instances.)
