Model Counting
==============

``aria.counting`` provides a shared frontend for counting satisfying assignments
from formulas and files, with theory-specific backends for Boolean, bit-vector,
and arithmetic fragments.

Current package layout
----------------------

Relevant pieces of the current package include:

* ``aria.counting.api``: dispatch and theory detection
* ``aria.counting.bool``: DIMACS and Boolean counting backends
* ``aria.counting.bv``: QF_BV model counting
* ``aria.counting.arith``: arithmetic counting, including LattE-based flows
* ``aria.counting.core``: structured result objects

Public entrypoints
------------------

The package exports these helpers:

.. code-block:: python

   from aria.counting import count, count_from_file, count_result, CountResult

``count()`` and ``count_result()`` operate on parsed Z3 formulas.
``count_from_file()`` and ``count_result_from_file()`` operate on ``.cnf``,
``.dimacs``, or ``.smt2`` files.

Formula-level counting
----------------------

.. code-block:: python

   import z3
   from aria.counting import count

   x = z3.Bool("x")
   y = z3.Bool("y")
   formula = z3.Or(x, y)

   num_models = count(formula)
   print(num_models)

The frontend detects the theory automatically unless ``theory=...`` is passed
explicitly.

File-based counting
-------------------

.. code-block:: python

   from aria.counting import count_from_file

   num_models = count_from_file("formula.cnf")
   print(num_models)

For richer metadata, use ``count_result()`` or ``count_result_from_file()``.

Supported counting modes
------------------------

* **Boolean**: DIMACS counting and Boolean SMT counting
* **Bit-vector**: ``aria.counting.bv`` backends
* **Arithmetic**: exact or backend-specific arithmetic counting when supported

When a theory or projection mode is unsupported, the structured result helpers
return an informative status instead of silently guessing.

CLI access
----------

The command-line frontend is:

.. code-block:: bash

   aria-mc formula.smt2
   python -m aria.cli.mc_cli formula.cnf --theory bool

See :doc:`../cli-tools/cli` for the current CLI surface.

Notes
-----

Some backends rely on optional external tools such as ``sharpSAT`` or LattE.
Availability depends on the selected theory and method.
