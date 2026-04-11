Monadic Predicate Abstraction
=============================

``aria.monabs`` provides monadic predicate-abstraction utilities for checking,
for each predicate in a set, whether it is compatible with a given formula.

Package layout
--------------

The current package includes:

* ``aria.monabs.cores``: core algorithms
* ``aria.monabs.analysis``: supporting analysis code
* ``aria.monabs.utils``: utilities
* ``aria.monabs.cpp``: C++ implementation and build example

Current API note
----------------

Older examples often referenced outdated top-level module locations. The current
code lives under ``aria.monabs.cores``.

Example imports
---------------

.. code-block:: python

   from aria.monabs.cores.unary_check import unary_check
   from aria.monabs.cores.dis_check import disjunctive_check_cached

Overview
--------

Given a formula ``F`` and predicates ``P1, ..., Pn``, monadic predicate
abstraction determines, for each predicate, whether ``F`` together with that
predicate is satisfiable.

Applications mentioned in the current package README include k-induction,
optimization/symbolic-abstraction support, and value-flow style analyses.
