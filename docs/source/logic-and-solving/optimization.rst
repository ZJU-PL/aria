Optimization Modulo Theory
==========================

``aria.pyomt`` contains optimization and MaxSAT components. It is no
longer described as a thin wrapper around an external ``pyomt`` dependency; the
repo contains its own optimization package and CLI frontend.

Current layout
--------------

The package currently contains:

* ``maxsmt/``: MaxSAT solvers and result handling
* ``omt_solver.py``: main OMT solver entrypoint
* ``omtarith/``: arithmetic OMT
* ``omtbv/``: bit-vector OMT
* ``omtfp/``: floating-point OMT
* ``msa/``: minimal satisfying assignment components
* ``omt_parser.py``, ``pysmt_utils.py``, ``bin_solver.py``: shared utilities

Floating-point OMT
------------------

The floating-point backends compare objectives with Z3's IEEE-754 numeric
relations (``fpLT``, ``fpLEQ``, ``fpGT``, ``fpGEQ``) and explicitly exclude NaN
objectives, treating NaN as worse than any numeric value. They therefore
implement IEEE-754 numeric ordering with NaN guards, not IEEE-754
``totalOrder``: ``+0.0`` and ``-0.0`` compare equal, and all NaNs are
equivalent to each other.

CLI access
----------

The main command-line frontend is:

.. code-block:: bash

   aria-pyomt problem.smt2
   python -m aria.cli.pyomt_cli problem.smt2 --engine qsmt

Related MaxSAT workflows are available through ``aria-maxsat``.

Public API note
---------------

``aria.pyomt`` currently exports result types such as
``OptimizationResult`` and ``OptimizationStatus``. Many solver implementations
live in subpackages such as ``aria.pyomt.maxsmt`` and the ``omt*``
directories.
