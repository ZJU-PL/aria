Interpolant Generation
======================

``aria.interpolant`` contains Craig-interpolant generation helpers for SMT
formulas.

Overview
--------

Given formulas ``A`` and ``B`` such that ``A ∧ B`` is unsatisfiable, a Craig
interpolant ``P`` satisfies:

* ``A`` implies ``P``
* ``P ∧ B`` is unsatisfiable
* ``P`` mentions only symbols shared by ``A`` and ``B``

Current components
------------------

The current package includes:

* ``pysmt_interpolant.py``: PySMT-based interpolation
* ``smtinterpol_interpolant.py``: SMTInterpol-based interpolation
* ``cvc5_interpolant.py``: cvc5-based interpolation

Typical use cases include:

* program verification
* predicate abstraction
* CEGAR refinement loops
* model checking

Programmatic usage
------------------

.. code-block:: python

   from aria.interpolant import pysmt_interpolant

   interpolant = pysmt_interpolant.get_interpolant(A, B)
