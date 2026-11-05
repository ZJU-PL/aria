SMT Solving
===========

``aria.smt`` collects SMT-oriented subpackages aimed at different theories and
solving strategies.

Current subpackages
-------------------

The current ``aria.smt`` tree includes:

* ``adt``: algebraic datatype solving
* ``arith``: arithmetic reasoning and related helpers
* ``bv``: bit-vector infrastructure and frontends
* ``bwind``: bit-width-independence solving
* ``ff``: finite-field SMT solvers and tooling
* ``fol``: first-order logic theorem proving (Miniprover)
* ``fp``: floating-point procedures and reductions
* ``lia_star``: LIA* and related BAPA-style support
* ``mba``: mixed Boolean-arithmetic simplification
* ``pcdclt``: parallel CDCL(T) stack
* ``portfolio``: QF_BV portfolio runner
* ``simplify``: formula simplification passes
* ``unification``: first-order unification and pattern matching
* ``unknown_resolver``: workflows for resolving ``unknown`` outcomes

Navigation
----------

This page is a package-level overview. For adjacent user-facing workflows, see:

* :doc:`parallel_cdclt` for parallel CDCL(T)
* :doc:`ff` for finite-field solving
* :doc:`fol` for the first-order theorem prover
* :doc:`unification` for term unification
* :doc:`optimization` for optimization-oriented stacks built on top of SMT
