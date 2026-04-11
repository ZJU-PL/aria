AllSMT
======

AllSMT extends AllSAT-style enumeration to SMT formulas. In ARIA, the current
user-facing API lives in ``aria.allsmt``.

Current API
-----------

The package exposes a factory-based interface:

* ``create_allsmt_solver()``
* ``AllSMTSolver``
* backends for Z3, PySMT, and MathSAT when available

Example
-------

.. code-block:: python

   from z3 import And, Ints
   from aria.allsmt import create_allsmt_solver

   x, y = Ints("x y")
   solver = create_allsmt_solver("z3")
   models = solver.solve(And(x + y == 5, x > 0, y > 0), [x, y], model_limit=10)

Common uses
-----------

* exhaustive test-input generation
* model enumeration for analysis
* projected reasoning workflows
* integration with verification and synthesis experiments

CLI access
----------

Use the CLI frontend for file-based workflows:

.. code-block:: bash

   aria-allsmt formula.smt2 --limit 50
   python -m aria.cli.allsmt_cli formula.smt2 --solver z3

For more detail, see ``aria/allsmt/README.md`` and :doc:`../cli-tools/cli`.
