Optimization Modulo Theory
==========================

``aria.optimization`` contains optimization and MaxSAT components. It is no
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

The floating-point stack uses IEEE-754 ``totalOrder`` semantics, so optimization
is defined over exact floating-point encodings rather than only the partial
numeric order induced by ``fp.lt`` or ``fp.leq``.

CLI access
----------

The main command-line frontend is:

.. code-block:: bash

   aria-pyomt problem.smt2
   python -m aria.cli.pyomt_cli problem.smt2 --engine qsmt

Related MaxSAT workflows are available through ``aria-maxsat``.

Public API note
---------------

``aria.optimization`` currently exports result types such as
``OptimizationResult`` and ``OptimizationStatus``. Many solver implementations
live in subpackages such as ``aria.optimization.maxsmt`` and the ``omt*``
directories.
