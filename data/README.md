# Data Directory

This directory contains benchmark files and test cases for the EFMC project.
The benchmark files are primarily in SMT2 and SyGuS(Inv) format.


## Boogie

## BV/ (Bit-Vector Benchmarks)
Contains bit-vector related benchmark files organized in subdirectories:
- `multi-phase/` - Multi-phase verification benchmarks
  - `sygus/` - SyGuS format benchmarks
  - `chc/` - Constrained Horn Clause benchmarks
- `LoopInvGen/` - Loop invariant generation benchmarks  
- `CAV19/` - Benchmarks from CAV 2019 conference

## C

C programs. For more details, please refer to `C/README.md`

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

## INT (Integer Benchmarks)
Contains integer arithmetic benchmark files:
- `multi-phase/` - Multi-phase verification benchmarks for integers
- `sygus-inv/` - SyGuS invariant synthesis benchmarks

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