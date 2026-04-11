Welcome to aria's Documentation!
================================

Introduction
------------

ARIA (Automated Reasoning Infrastructure & Applications) is a Python toolkit
and research playground for automated reasoning. The repository combines
libraries, CLI tools, and research prototypes across SAT/SMT solving,
quantified reasoning, model counting, optimization, theorem proving, program
verification, symbolic abstraction, and related experimentation.

Major user-facing areas include:

* ``aria.bool``: SAT, MaxSAT, QBF, CNF simplification, and knowledge compilation
* ``aria.smt``: SMT-oriented theory packages and solver utilities
* ``aria.srk``: symbolic reasoning infrastructure
* ``aria.quant``: quantified reasoning, EFSMT, QE, CHC tooling, and prototypes
* ``aria.efmc``: program verification frontends and engines
* ``aria.counting`` / ``aria.sampling`` / ``aria.prob``: counting, sampling,
  and probabilistic reasoning
* ``aria.optimization``: optimization and MaxSMT/MaxSAT-oriented workflows
* ``aria.itp`` / ``aria.fol`` / ``aria.interpolant``: theorem proving and logic
* ``aria.translator`` / ``aria.cli``: translators and command-line tools

Installing and using aria
-------------------------

Install from source:

.. code-block:: bash

   git clone https://github.com/ZJU-PL/aria
   cd aria
   bash setup_local_env.sh
   pip install -e .

For a faster local setup with ``uv``:

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install -e .

Common console commands after installation include:

* ``aria-fmldoc``
* ``aria-mc``
* ``aria-pyomt``
* ``aria-efsmt``
* ``aria-maxsat``
* ``aria-unsat-core``
* ``aria-allsmt``
* ``aria-smt-server``
* ``aria-efmc``
* ``aria-efmc-efsmt``
* ``aria-polyhorn``

Documentation map
-----------------

* ``getting-started``: installation notes, quick reference, and tutorials
* ``logic-and-solving``: core solver and reasoning packages
* ``proofs-and-explanations``: abduction, interpolation, and theorem proving
* ``quantified-reasoning``: EFSMT, QE, and quantified workflows
* ``verification`` / ``abstraction``: EFMC engines and abstraction-oriented pages
* ``counting-probability``: counting, sampling, and probabilistic reasoning
* ``cli-tools``: command-line frontends and translator-oriented workflows

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    getting-started/index
    logic-and-solving/index
    proofs-and-explanations/index
    quantified-reasoning/index
    automata-languages/index
    verification/index
    abstraction/index
    counting-probability/index
    synthesis/index
    logic-programming/index
    llm-ml/index
    cli-tools/index
    global_params
