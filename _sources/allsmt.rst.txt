AllSMT
===========================


=====================
Introduction
=====================
AllSMT (All-Solutions Satisfiability Modulo Theories) extends AllSAT to SMT formulas,
aiming to enumerate all satisfying assignments of an SMT formula. This problem is
fundamental in various applications, including:

* Combinational iteration testing
* Quantitative program analysis
* Constrained test generation
* Program synthesis
* Formal verification


=====================
Related Work
=====================

The closely related AllSAT problem (finding all satisfying assignments for Boolean formulas)
serves as a foundation for AllSMT. AllSAT is NP-hard and not fixed-parameter tractable
unless P=NP.

Current AllSAT approaches fall into two categories:

Blocking Solvers
~~~~~~~~~~~~~~~
* Built on CDCL (Conflict-Driven Clause Learning)
* Uses non-chronological backtracking (NCB)
* Adds blocking clauses to prevent duplicate solutions

Non-blocking Solvers
~~~~~~~~~~~~~~~~~~~
* Avoids blocking clauses overhead
* Employs chronological backtracking (CB)
* Generally more memory-efficient

AllSMT in Aria
--------------
[Content to be added]


=====================
References
=====================
.. [SAT23] Masina, G., Spallita, G., Sebastiani, R. (2023).
    *On CNF Conversion for Disjoint SAT Enumeration*. SAT 2023.

.. [TACAS05] Jin, H., Han, H., Somenzi, F. (2005).
    *Efficient Conflict Analysis for Finding All Satisfying Assignments of a Boolean Circuit*. TACAS 2005.

.. [CAV02] McMillan, K. L. (2002).
    *Applying SAT Methods in Unbounded Symbolic Model Checking*. CAV 2002.

.. [FMCAD04] Grumberg, O., Schuster, A., Yadgar, A. (2004).
    *Memory Efficient All-Solutions SAT Solver and Its Application for Reachability Analysis*. FMCAD 2004.

.. [DATE04] Li, B., Hsiao, M. S., Sheng, S. (2004).
    *A Novel SAT All-Solutions Solver for Efficient Preimage Computation*. DATE 2004.

.. [DPBS] Spallitta, G., Sebastiani, R., Biere, A.
    *Disjoint Projected Enumeration for SAT and SMT without Blocking Clauses*.
    `GitHub Repository <https://github.com/giuspek/tabularAllSAT>`_
