Program Verification
====================

``aria.efmc`` is the verification-oriented area of the repository. It combines
frontends, transition-system construction, and multiple proving engines.

Verification pipeline
---------------------

At a high level, EFMC follows this workflow:

1. Parse an input frontend such as CHC, SyGuS, Boogie, or a C-oriented flow.
2. Build an internal transition-system representation.
3. Select a proving engine.
4. Run the engine and report ``safe``, ``unsafe``, or ``unknown``.

Main engines
------------

Engines selectable through ``aria-efmc --engine`` (see ``run_verifier`` in
``aria/cli/efmc_cli.py``):

* **EF / template-based** invariant synthesis (``ef``)
* **PDR** via Z3's CHC/Spacer engine (``pdr``)
* **K-induction** (``kind``)
* **QE** and **QI** verification paths (``qe``, ``qi``)
* **Houdini** (``houdini``)
* **Abduction** (``abduction``)
* **BDD-based** verification (``bdd``)
* **Predicate abstraction** and **symbolic abstraction** (``predabs``,
  ``symabs``)
* **LLM4Inv** (``llm4inv``)

Additional engines that are only available through the Python API (they have
no ``--engine`` value):

* **Danger invariants** -- ``aria.efmc.engines.danger.DangerInvariantProver``
* **k-safety / hyperproperties** --
  ``aria.efmc.engines.ksafety`` (``NonInterferenceProver``,
  ``HyperLTLProver``, and related classes)

PolyHorn is a separate solver with its own frontend and command-line entrypoint
(``aria-polyhorn``); it is not selectable from ``aria-efmc``.

CLI entrypoints
---------------

Primary verification commands:

.. code-block:: bash

   aria-efmc --help
   aria-efmc-efsmt --help
   aria-polyhorn --help

Equivalent module entrypoints:

.. code-block:: bash

   python -m aria.cli.efmc_cli --help
   python -m aria.cli.efmc_efsmt_cli --help
   python -m aria.cli.polyhorn_cli --help

Where to look next
------------------

* :doc:`pdr`
* :doc:`smttools`
* :doc:`kinduction`
* :doc:`houdini`
* :doc:`ksafety`
* :doc:`llm4inv`
* :doc:`polyhorn`

For exact engine flags and supported options, prefer the current CLI help and
the parser definitions in ``aria/cli/efmc_cli.py``.
