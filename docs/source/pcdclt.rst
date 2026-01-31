Parallel SMT CDCL(T) Solving
=================================


==========
CDCL(T)
==========


CDCL(T) stands for Conflict-Driven Clause Learning modulo Theory, which is a powerful
algorithm for solving Satisfiability Modulo Theories (SMT) problems.
It extends the CDCL algorithm used in SAT solving to handle theories beyond
pure propositional logic.

- Combines boolean reasoning (SAT) with theory reasoning
- Maintains a boolean abstraction of the formula
- Performs theory-specific consistency checks
- Learns new clauses from conflicts



==========
Parallel CDCL(T)
==========


==========
How to Use
==========


1. Build and install all binary solvers in the ``bin_solvers`` directory.
2. Install the package and its dependencies: ``pip install -e .`` (see ``pyproject.toml``).


============
Related Work
=============


Refer to the following links for related work:

- [https://smt-comp.github.io/2022/results/results-cloud](https://smt-comp.github.io/2022/results/results-cloud)
- [https://github.com/usi-verification-and-security/SMTS/tree/cube-and-conquer-fixed](https://github.com/usi-verification-and-security/SMTS/tree/cube-and-conquer-fixed)
- [https://github.com/usi-verification-and-security/SMTS/tree/portfolio](https://github.com/usi-verification-and-security/SMTS/tree/portfolio)
- cvc5-cloud: [https://github.com/amaleewilson/aws-satcomp-solver-sample/tree/cvc5](https://github.com/amaleewilson/aws-satcomp-solver-sample/tree/cvc5)
- Vampire portfolio: [https://smt-comp.github.io/2022/system-descriptions/Vampire.pdf](https://smt-comp.github.io/2022/system-descriptions/Vampire.pdf)


Refer to https://smtlib.cs.uiowa.edu/benchmarks.shtml for benchmarks.

