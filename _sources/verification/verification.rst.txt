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

Current engine families include:

* **EF / template-based** invariant synthesis
* **PDR** via Spacer-oriented workflows
* **K-induction**
* **Houdini**
* **Abduction**
* **QE** and **QI** verification paths
* **Predicate abstraction** and **symbolic abstraction**
* **BDD-based** verification
* **LLM4Inv**
* specialized areas such as **PolyHorn**, **k-safety**, **danger**, and
  abstract-interpretation-related flows

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

* :doc:`kinduction`
* :doc:`houdini`
* :doc:`ksafety`
* :doc:`llm4inv`
* :doc:`polyhorn`

For exact engine flags and supported options, prefer the current CLI help and
the parser definitions in ``aria/cli/efmc_cli.py``.
