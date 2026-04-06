Machine Learning Components
===========================

The ``aria.ml`` package collects machine-learning- and LLM-oriented research
components for automated reasoning. It is best understood as a toolbox of
subsystems rather than a single unified API.

Overview
--------

The main areas in ``aria/ml`` are:

- ``llm/``: LLM-assisted reasoning utilities, including natural-language
  interfaces, abduction, trigger generation, and solver-aided workflows.
- ``smtgazer/``: machine-learning-based SMT solver portfolio selection.
- ``machfea/``: feature extraction for SMT instances.
- ``tactic_opt/``: genetic-algorithm search for effective Z3 tactic sequences.

LLM-Assisted Reasoning
----------------------

``aria/ml/llm`` contains several LLM-facing components:

- ``smt2nl.py`` converts SMT-LIB assertions into natural language.
- ``abduction/`` provides natural-language abduction workflows and result data
  structures.
- ``ematching/`` provides trigger generation and trigger-selection helpers for
  quantified SMT formulas, including a CLI-oriented entry point.
- ``smto/`` contains solver-aided reasoning utilities for synthesizing or using
  specifications of closed-box functions.
- ``induction/`` contains experimental LLM-guided induction workflows and
  benchmark-driving scripts.

Representative imports include:

.. code-block:: python

   from aria.ml.llm.abduction import NLAbductor
   from aria.ml.llm.ematching import LLMTriggerGenerator, TriggerSelector
   from aria.ml.llm.smto import PS_SMTOSolver

There is also a simple command-line style module for SMT-to-natural-language
conversion:

.. code-block:: bash

   python -m aria.ml.llm.smt2nl "(assert (and (> x 5) (<= y 10)))"

Portfolio Selection and Feature Extraction
------------------------------------------

``aria/ml/smtgazer`` and ``aria/ml/machfea`` support ML-guided solver
selection for SMT workloads.

- ``smtgazer/`` contains the main portfolio-training and evaluation scripts,
  including ``SMTportfolio.py``, ``batchportfolio.py``, and
  ``portfolio_smac3.py``.
- ``machfea/`` provides problem-feature extraction and batch inference helpers,
  including ``get_feature.py`` and ``mach_run_inference.py``.

These components are script-heavy and research-oriented, so they are most
useful when you want to reproduce experiments or build a solver-scheduling
pipeline over benchmark datasets.

Tactic Optimization
-------------------

``aria/ml/tactic_opt`` searches for good Z3 tactic sequences using a genetic
algorithm. The most visible package is ``ga_tactics``, which exports the core
objects for modeling, evaluating, and evolving tactic sequences.

Representative imports include:

.. code-block:: python

   from aria.ml.tactic_opt.ga_tactics import GA, TacticSeq

Examples and Entry Points
-------------------------

Useful starting points include:

- ``aria.ml.llm.abduction.NLAbductor``
- ``aria.ml.llm.ematching.LLMTriggerGenerator``
- ``aria.ml.llm.ematching.TriggerSelector``
- ``aria.ml.llm.smto.PS_SMTOSolver``
- ``aria.ml.tactic_opt.ga_tactics.GA``
- ``python -m aria.ml.llm.smt2nl``

Status Notes
------------

``aria.ml`` mixes reusable library code with experiment-oriented scripts.
Some subdirectories are better viewed as paper or benchmark artifacts than as
stable end-user APIs, especially under ``llm/induction`` and the portfolio
training workflows.

Further Reading
---------------

- ``aria/ml/README.md`` for the package-level overview
- ``aria/ml/llm/induction/README.md`` for the induction workflow
- ``aria/ml/smtgazer/README.md`` for portfolio experiments
- ``aria/ml/tactic_opt/ga_tactics/README.md`` for tactic optimization details
