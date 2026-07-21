Shared-Context Batched Satisfiability
======================================

``aria.scsat`` provides shared-context batched satisfiability utilities for
checking, for each predicate in a set, whether it is compatible with a given
formula.

Package layout
--------------

The current package includes:

* ``aria.scsat.cores``: core algorithms
* ``aria.scsat.analysis``: supporting analysis code
* ``aria.scsat.utils``: utilities
* ``aria.scsat.cpp``: C++ implementation and build example

Current API note
----------------

Older examples often referenced outdated top-level module locations. The current
code lives under ``aria.scsat.cores``.

Example imports
---------------

.. code-block:: python

   from aria.scsat.cores.unary_check import unary_check
   from aria.scsat.cores.dis_check import disjunctive_check_cached

Overview
--------

Given a formula ``F`` and predicates ``P1, ..., Pn``, shared-context batched
satisfiability determines, for each predicate, whether ``F`` together with that
predicate is satisfiable.

Applications mentioned in the current package README include k-induction,
optimization/symbolic-abstraction support, and value-flow style analyses.
