Houdini
=======

``aria.efmc.engines.houdini`` contains Houdini-style invariant pruning.

Current engine surface
----------------------

The package currently exposes:

* ``HoudiniProver``

It works with a transition system plus a candidate predicate/template set and
iteratively removes predicates that fail the inductiveness checks.

How it fits in EFMC
-------------------

Houdini is a lightweight invariant-inference technique:

1. start from a conjunction of candidate predicates
2. check inductiveness against the transition relation
3. remove predicates that are disproved
4. repeat until the remaining conjunction is inductive or exhausted

Programmatic usage
------------------

.. code-block:: python

   from aria.efmc.engines.houdini import HoudiniProver
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem(...)
   prover = HoudiniProver(sts)
   result = prover.solve()

   # Candidate lemmas are derived from the post-condition (plus
   # ``sts.invariants`` when present). To prune an explicit set instead:
   # lemmas = [x >= 0, x + y == 5, ...]
   # kept = prover.houdini(lemmas)  # surviving candidates keyed by index

CLI usage
---------

Houdini-related workflows are exposed through the main verifier CLI:

.. code-block:: bash

   aria-efmc --lang chc --engine houdini --file program.smt2

For exact CLI flags and current predicate/template handling, prefer
``aria-efmc --help`` and ``aria/efmc/engines/houdini/``.
