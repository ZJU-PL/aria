Optimization Modulo Theory
=============================

============
Introduction
============

Optimization Modulo Theories (OMT) extends Satisfiability Modulo Theories (SMT)
by adding optimization capabilities. While SMT focuses on finding satisfying
assignments to formulas in first-order logic with respect to background theories,
OMT additionally allows for finding optimal solutions according to objective
functions.


=============
OMT in Aria
=============

Currently, we rely on the ``pyomt`` library for solving OMT problems. (TBD: need
to add it as a requirement)

Basic Usage


.. code-block:: python
    from


===========
References
===========

- Sebastiani, R., & Trentin, P. (2015). "Optimization Modulo Theories with Linear Rational Costs". ACM Transactions on Computational Logic.
- Li, Y., Albarghouthi, A., Kincaid, Z., Gurfinkel, A., & Chechik, M. (2014). "Symbolic Optimization with SMT Solvers". POPL.
- Bjørner, N., Phan, A. D., & Fleckenstein, L. (2015). "νZ - An Optimizing SMT Solver". TACAS.
