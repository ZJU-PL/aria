Quick Reference
===============

This page gives a compact reference to the current high-level ARIA entrypoints.

Installation
------------

.. code-block:: bash

   pip install aria

For local development:

.. code-block:: bash

   git clone https://github.com/ZJU-PL/aria
   cd aria
   pip install -e .

Useful Python entrypoints
-------------------------

Model counting
~~~~~~~~~~~~~~

.. code-block:: python

   import z3
   from aria.counting import count, count_from_file

   x = z3.Bool("x")
   y = z3.Bool("y")
   print(count(z3.Or(x, y)))
   print(count_from_file("formula.cnf"))

AllSMT
~~~~~~

.. code-block:: python

   from z3 import And, Ints
   from aria.allsmt import create_allsmt_solver

   x, y = Ints("x y")
   solver = create_allsmt_solver("z3")
   models = solver.solve(And(x + y == 5, x > 0, y > 0), [x, y], model_limit=10)

Backbone computation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.backbone import compute_backbone
   from aria.backbone import BackboneAlgorithm

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
   options = SamplingOptions(method=SamplingMethod.ENUMERATION, num_samples=5)
   result = sample_models_from_formula(
       z3.And(x + y > 0, x - y < 1),
       Logic.QF_LRA,
       options,
   )

Unsat cores
~~~~~~~~~~~

.. code-block:: python

   from aria.unsat_core import Algorithm, UnsatCoreComputer

Abduction and belief revision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.abduction import abduce, revise_belief_base

Verification and CLI workflows
------------------------------

Common command-line entrypoints:

.. code-block:: bash

   aria-fmldoc --help
   aria-mc --help
   aria-pyomt --help
   aria-efsmt --help
   aria-efmc --help
   aria-polyhorn --help

Equivalent module entrypoints:

.. code-block:: bash

   python -m aria.cli.fmldoc_cli --help
   python -m aria.cli.mc_cli --help
   python -m aria.cli.pyomt_cli --help
   python -m aria.cli.efsmt_cli --help
   python -m aria.cli.efmc_cli --help
   python -m aria.cli.polyhorn_cli --help

Package map
-----------

* ``aria.bool``: Boolean reasoning and MaxSAT helpers
* ``aria.smt``: SMT package families
* ``aria.quant``: quantified reasoning
* ``aria.efmc``: verification frontends and engines
* ``aria.symabs`` / ``aria.monabs``: abstraction-oriented packages
* ``aria.translator``: format conversion

For package-specific examples, prefer the nearest package ``README.md`` and the
section pages linked from this documentation tree.
