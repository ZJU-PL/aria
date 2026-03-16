Abstraction Interpretation
===========================

Abstraction interpretation is a formal framework for static analysis of programs.
It is based on the idea of abstracting the concrete semantics of a program to
a simpler, more tractable abstract semantics. This abstraction is then used to reason about the program's behavior. The key idea is to over-approximate the set of possible
behaviors of the program, so that the analysis is sound but not necessarily
complete.


===========================
Chaotic Iteration
===========================

The basic idea of abstraction interpretation is to iteratively apply an abstract
transformer to an abstract state until a fixpoint is reached. The abstract
transformer is a function that maps an abstract state to another abstract state.
The fixpoint is reached when the abstract state does not change anymore. 
This process is often called chaotic iteration, and it is guaranteed to converge to a fixpoint

Widening
========

Widening is a technique used in abstract interpretation to ensure the convergence of the iterative analysis process. When the abstract state space is infinite or very large, the analysis might not reach a fixpoint in a reasonable amount of time. Widening accelerates the convergence by forcing the iteration to jump to a broader abstract state, effectively over-approximating the union of the states encountered during the iteration. This guarantees that the analysis will terminate, although it may result in a less precise fixpoint.

Lookahead Widening
----------------------

Lookahead widening is a technique used in abstract interpretation to improve the precision of the analysis. It works by looking ahead at the next few iterations of the analysis and using that information to make a more informed decision about the current abstract state.


Guided Static Analysis
-----------------------

Widening with Thresholds
------------------------



Policy Iteration
-----------------------



===========================
Systematic Abstraction
===========================

Making Abstraction Interpretation Complete
==========================================




=============================
Abstract Interpretation Tools
=============================

- Astree: developed by the French Space Agency (CNES) for the verification of critical embedded software.

- FramaC

- PolySpace: by MathWorks for the verification of embedded software.

- CodeSonar: by GrammaTech

- TVLA: by University of Wisconsin-Madison, Tel Aviv University, etc.

- IKOS: by NASA 

- Crab

===========================
References
===========================

.. [CC77] P. Cousot and R. Cousot. Abstract interpretation: A unified lattice model for static analysis of programs by construction or approximation of fixpoints. POPL'97.
.. [CC92] P. Cousot and R. Cousot. Abstract interpretation frameworks. Journal of Logic and Computation, 2(4):511-547, 1992.
.. [Reps94] T. Reps. Solving demand versions of interprocedural analysis problems.
.. [Reps97] T. Reps, S. Horwitz, and M. Sagiv. Precise interprocedural dataflow analysis via graph reachability, POPL'95.
.. [Reps98] T. Reps, S. Horwitz, M. Sagiv, and G. Rosay. Speeding up slicing. FSE'??
.. [xx] A. Adje, S. Gaubert, and E. Goubault. Coupling Policy Iteration with Semidefinite Relaxation to Compute Accurate Numerical Invariants in Static
Analysis. ESOP'10.
.. [xx] A. Costan, S. Gaubert, E. Goubault, M. Martel, and S. Putot. A Policy Iteration Algorithm for Computing Fixed Points in Static Analysis of
Programs. CAV'05.
