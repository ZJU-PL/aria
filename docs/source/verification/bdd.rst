BDD-Based Verification
======================

``aria.efmc.engines.bdd`` currently exposes ``BDDProver`` for symbolic
reachability-style verification over Boolean-flavored transition systems.

Current API
-----------

.. code-block:: python

   from aria.efmc.engines.bdd import BDDProver
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem(...)
   prover = BDDProver(sts, use_forward=True, max_iterations=1000)
   result = prover.solve()

High-level behavior
-------------------

The prover supports forward and backward style reachability computations and is
documented in the codebase as a BDD-oriented symbolic model-checking path within
EFMC.

Notes
-----

This page is intentionally package-focused rather than a full BDD tutorial.
For current behavior and tunable parameters, check
``aria/efmc/engines/bdd/bdd_prover.py`` and the EFMC CLI help.
