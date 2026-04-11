Tutorial
========

This tutorial is a current high-level orientation to ARIA's package layout and
entrypoints. The repository evolves quickly, so package README files and CLI
``--help`` output are the most reliable detailed references.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install aria

Or install the development version from source:

.. code-block:: bash

   git clone https://github.com/ZJU-PL/aria
   cd aria
   bash setup_local_env.sh
   pip install -e .

Main package areas
------------------

ARIA is organized around several major user-facing subsystems:

* ``aria.bool``: SAT, MaxSAT, QBF, CNF simplification, and compilation
* ``aria.smt``: SMT-oriented theory packages and solver utilities
* ``aria.srk``: symbolic reasoning infrastructure
* ``aria.quant``: EFSMT, QE, CHC tooling, and quantified-reasoning prototypes
* ``aria.efmc``: verification frontends and engines
* ``aria.counting`` / ``aria.sampling`` / ``aria.prob``: counting and sampling
* ``aria.optimization``: optimization and MaxSMT-related flows
* ``aria.translator`` / ``aria.cli``: format translation and command-line tools

Current CLI workflow
--------------------

Installed console commands include:

.. code-block:: bash

   aria-fmldoc
   aria-mc
   aria-pyomt
   aria-efsmt
   aria-maxsat
   aria-unsat-core
   aria-allsmt
   aria-smt-server
   aria-efmc
   aria-efmc-efsmt
   aria-polyhorn

You can also invoke the corresponding Python modules directly:

.. code-block:: bash

   python -m aria.cli.mc_cli --help
   python -m aria.cli.efmc_cli --help

Examples of current Python APIs
-------------------------------

Model counting
~~~~~~~~~~~~~~

.. code-block:: python

   import z3
   from aria.counting import count

   x = z3.Bool("x")
   y = z3.Bool("y")
   print(count(z3.Or(x, y)))

AllSMT enumeration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from z3 import And, Ints
   from aria.allsmt import create_allsmt_solver

   x, y = Ints("x y")
   solver = create_allsmt_solver("z3")
   solver.solve(And(x + y == 5, x > 0, y > 0), [x, y], model_limit=10)

Sampling
~~~~~~~~

.. code-block:: python

   import z3
   from aria.sampling import (
       Logic,
       SamplingMethod,
       SamplingOptions,
       sample_models_from_formula,
   )

   x, y = z3.Reals("x y")
   sample_models_from_formula(
       z3.And(x + y > 0, x - y < 1),
       Logic.QF_LRA,
       SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=3),
   )

Unsat-core extraction
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.unsat_core import Algorithm, UnsatCoreComputer

Verification with EFMC
----------------------

The main verification frontend is ``aria-efmc`` / ``python -m aria.cli.efmc_cli``.
It works over frontends such as CHC, SyGuS, Boogie, and C-oriented workflows and
selects among engines including EF-template solving, PDR, k-induction, Houdini,
abduction, predicate abstraction, symbolic abstraction, and LLM4Inv.

Start with:

.. code-block:: bash

   aria-efmc --help

Where to go next
----------------

* :doc:`quickref` for a compact API and CLI cheat sheet
* :doc:`verification-tutorial` for EFMC-oriented usage
* package ``README.md`` files under ``aria/`` for subsystem-specific details
