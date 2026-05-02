PDR (Property-Directed Reachability)
=====================================

``aria.efmc.engines.pdr`` provides a Property-Directed Reachability (IC3)
verification engine for transition systems.

Current engine surface
----------------------

The package exposes:

* ``PDRProver``

It operates over ``aria.efmc.sts.TransitionSystem`` and returns results
through ``VerificationResult``.

Idea
----

PDR (also known as IC3) proves safety by searching for an inductive invariant
that is reachable from the initial states, preserved by transitions, and
disjoint from unsafe states. Rather than implementing IC3 from scratch, the
current engine encodes the verification problem as Constrained Horn Clauses
(CHC) and delegates the PDR search to Z3's ``HORN`` solver.

The encoding constructs three Horn clauses over an uninterpreted predicate
``inv``:

* **Init**: ``∀ vars. init(vars) ⟹ inv(vars)``
* **Inductive**: ``∀ vars, vars'. inv(vars) ∧ trans(vars, vars') ⟹ inv(vars')``
* **Post**: ``∀ vars. inv(vars) ⟹ post(vars)``

When Z3 finds a satisfying model, ``inv`` is extracted as the inductive
invariant. Supported variable sorts include ``Int``, ``Real``, ``BitVec``,
and ``Bool``.

Programmatic usage
------------------

.. code-block:: python

   from aria.efmc.engines.pdr import PDRProver
   from aria.efmc.sts import TransitionSystem

   sts = TransitionSystem(...)
   prover = PDRProver(sts)
   prover.set_verbose(True)
   result = prover.solve(timeout=60)

   if result.is_safe:
       print("Invariant:", result.invariant)
   elif result.is_unknown:
       print("Unknown (may have timed out)")

CLI usage
---------

Use the EFMC frontend:

.. code-block:: bash

   aria-efmc --lang chc --engine pdr --file program.smt2

For exact flags see ``aria.cli.efmc_cli``.
