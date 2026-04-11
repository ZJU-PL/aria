Program Verification with EFMC
==============================

This page introduces the current EFMC verification workflow in ARIA.

Setup
-----

Install ARIA locally:

.. code-block:: bash

   git clone https://github.com/ZJU-PL/aria
   cd aria
   pip install -e .

Then inspect the verifier CLI:

.. code-block:: bash

   aria-efmc --help
   python -m aria.cli.efmc_cli --help

Supported frontends
-------------------

The main EFMC CLI currently works with frontends such as:

* CHC
* SyGuS
* Boogie
* C-oriented frontend workflows

Supported engines
-----------------

Current engine families exposed by the verifier include:

* ``ef``: template-based invariant synthesis
* ``pdr``: Property-Directed Reachability
* ``kind``: k-induction
* ``qe`` / ``qi``: quantifier-based verification paths
* ``houdini``: conjunctive invariant pruning
* ``abduction``: abductive invariant generation
* ``bdd``: BDD-based verification
* ``predabs``: predicate abstraction
* ``symabs``: symbolic abstraction
* ``llm4inv``: LLM-guided invariant synthesis

Typical commands
----------------

.. code-block:: bash

   aria-efmc --lang chc --engine ef --file program.smt2
   aria-efmc --lang chc --engine pdr --file program.smt2
   aria-efmc --lang chc --engine kind --file program.smt2
   aria-efmc --lang chc --engine symabs --symabs-domain interval --file program.smt2
   aria-efmc --lang chc --engine llm4inv --file program.smt2

Results
-------

The verifier generally reports one of:

* ``safe``
* ``unsafe``
* ``unknown``

Related entrypoints
-------------------

Adjacent verification CLIs include:

* ``aria-efmc-efsmt`` / ``python -m aria.cli.efmc_efsmt_cli``
* ``aria-polyhorn`` / ``python -m aria.cli.polyhorn_cli``

Notes
-----

Exact option sets evolve with the verifier. For concrete workflows and current
flags, prefer ``aria-efmc --help`` and the parser definitions in
``aria/cli/efmc_cli.py``.
