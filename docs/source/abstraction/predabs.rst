Predicate Abstraction and CEGAR
===============================

This page covers the EFMC-facing predicate-abstraction engine rather than the
lower-level standalone symbolic-abstraction helpers documented elsewhere.

Current engine surface
----------------------

``aria.efmc.engines.predabs`` currently exposes ``PredicateAbstractionProver``.

At a high level, predicate abstraction restricts reasoning to Boolean
combinations of a chosen predicate set, producing a finite abstraction that can
be used in verification workflows.

Programmatic usage
------------------

.. code-block:: python

   from aria.efmc.engines.predabs import PredicateAbstractionProver
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem(...)
   prover = PredicateAbstractionProver(sts)
   result = prover.solve()

Relationship to other docs
--------------------------

* :doc:`predicate_abstraction` covers the lower-level package under
  ``aria.symabs.predicate_abstraction``
* this page is about the verification-engine view inside EFMC
