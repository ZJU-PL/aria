K-Induction
===========

``aria.efmc.engines.kinduction`` provides k-induction-based verification over
transition systems.

Current engine surface
----------------------

The package currently exposes:

* ``KInductionProver``
* ``KInductionProverInc``

Both operate over ``aria.efmc.sts.TransitionSystem`` and return verification
results through their ``solve(...)`` methods.

Idea
----

K-induction strengthens standard induction by checking a bounded base case and
an inductive step over ``k`` consecutive states. In verification workflows this
is useful when one-step induction is too weak to prove the target property.

Programmatic usage
------------------

.. code-block:: python

   from aria.efmc.engines.kinduction import KInductionProver
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem(...)
   prover = KInductionProver(sts)
   result = prover.solve(k=30)

CLI usage
---------

Use the EFMC frontend for file-based workflows:

.. code-block:: bash

   aria-efmc --lang chc --engine kind --file program.smt2

For exact flags and optional auxiliary-invariant settings, see
``aria.cli.efmc_cli`` and ``aria/efmc/engines/kinduction/``.
