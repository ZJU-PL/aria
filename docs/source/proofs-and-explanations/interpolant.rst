Interpolant Generation
======================

``aria.proof.interpolant`` contains Craig-interpolant generation helpers for SMT
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

   from aria.proof.interpolant import PySMTInterpolantSynthesizer

   # PySMT picks an available interpolating backend (MathSAT or OptiMathSAT)
   synth = PySMTInterpolantSynthesizer()
   # interpolant = synth.interpolate(A, B)

   # Or select one explicitly
   synth = PySMTInterpolantSynthesizer(solver_name="msat")

``PySMTInterpolantSynthesizer`` delegates to PySMT, so ``A ∧ B`` must be
unsatisfiable and an interpolation-capable backend must be installed; Z3 alone
is not sufficient.
