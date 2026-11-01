Unification
===========

Introduction
------------

The unification module (``aria.smt.unification``) provides a general-purpose term
unification engine supporting first-order unification, pattern matching, and
substitution operations.

Key Features
------------

* **First-Order Unification**: Robinson-style term unification algorithm
* **Pattern Matching**: Shape-based dispatch and term matching
* **Substitution Composition**: Efficient substitution management via ``reify``
* **Generic Dispatch**: Multiple dispatch for extensibility
* **Recursive Unification**: Deep structure unification with logic variables

Components
==========

Core Unification (``aria.smt.unification.core``)
------------------------------------------------

Main unification engine:

.. code-block:: python

   from aria.smt.unification import unify, reify, var

   # Define variables and terms
   x = var("x")
   y = var("y")

   # Unify terms
   substitution = unify((x, 2), (1, y))
   # Result: {x: 1, y: 2}

   # Reify terms under substitution
   res = reify((x, y), substitution)
   # Result: (1, 2)

Pattern Matching (``aria.smt.unification.match``)
-------------------------------------------------

Pattern-directed dispatch based on argument shape:

.. code-block:: python

   from aria.smt.unification import var
   from aria.smt.unification.match import Dispatcher

   x = var("x")
   d = Dispatcher("my_func")
   d.add((1, x), lambda a, b: f"matched 1 and {b}")

Variables (``aria.smt.unification.variable``)
---------------------------------------------

Variable representation and utilities:

.. code-block:: python

   from aria.smt.unification import Var, isvar, var

   # Create variables
   x = var("x")

   # Check if term is a logic variable
   print(isvar(x))  # True
   print(isvar(5))  # False

Utilities (``aria.smt.unification.core``)
-----------------------------------------

Helper functions for substitution:

.. code-block:: python

   from aria.smt.unification.core import walk

   # Walk through substitutions
   # result = walk(term, substitution)

Applications
============

* **Term Rewriting**: Equation solving and simplification
* **Logic Programming**: Rule matching and resolution
* **Type Inference**: Structural type unification
* **Constraint Solving**: Unification constraints in SMT

References
==========

- Robinson, J. A. (1965). *A Machine-Oriented Logic Based on the Resolution Principle*. JACM 1965
- Martelli, A., & Montanari, U. (1982). *An Efficient Unification Algorithm*. TOPLAS 1982
- Baader, F., & Snyder, W. (2001). *Unification Theory*. Handbook of Automated Reasoning
