Unification
===========================

Introduction
=====================

The unification module (``aria/unification``) provides a general-purpose term unification engine supporting first-order unification, pattern matching, and substitution operations. It is used throughout aria for term manipulation, pattern matching in theorem proving, and constraint solving.

Key Features
-------------

* **First-Order Unification**: Robinson's unification algorithm
* **Pattern Matching**: One-way matching for rewriting
* **Substitution Composition**: Efficient substitution management
* **Generic Dispatch**: Type-based dispatch for extensibility
* **Recursive Unification**: Deep structure unification with occurs check

Components
=====================

Core Unification (``aria/unification/core.py``)
-------------------------------------------------

Main unification engine:

.. code-block:: python

   from aria.unification import unify, Var

   # Define variables and terms
   x = Var('x')
   y = Var('y')

   # Unify terms
   substitution = unify((x, 2), (1, y))
   # Result: {x: 1, y: 2}

   # Unify complex structures
   s = unify([x, [y, 3]], [1, [2, 3]])
   # Result: {x: 1, y: 2}

Pattern Matching (``aria/unification/match.py``)
--------------------------------------------------

One-way pattern matching for term rewriting:

.. code-block:: python

   from aria.unification import match, Var

   # Match pattern against term
   pattern = (Var('x'), Var('y'), Var('x'))
   term = (1, 2, 1)

   result = match(pattern, term)
   # Result: {x: 1, y: 2}

Variables (``aria/unification/variable.py``)
----------------------------------------------

Variable representation and utilities:

.. code-block:: python

   from aria.unification import Var, isvar

   # Create variables
   x = Var('x')
   y = Var('y')

   # Check if term is variable
   isvar(x)  # True
   isvar(5)  # False

Utilities (``aria/unification/utils.py``)
-------------------------------------------

Helper functions for substitution and term manipulation:

.. code-block:: python

   from aria.unification.utils import walk, occurs_check

   # Walk through substitutions
   result = walk(term, substitution)

   # Check for circular references
   valid = occurs_check(var, term, substitution)

Type Dispatch (``aria/unification/dispatch.py``)
--------------------------------------------------

Generic function dispatch based on type:

.. code-block:: python

   from aria.unification.dispatch import dispatch

   @dispatch(int, int)
   def unify_ints(x, y):
       return {x: y} if x != y else {}

Applications
=====================

* **Theorem Proving**: Pattern matching in the ITP module
* **Term Rewriting**: Equation solving and simplification
* **Logic Programming**: Prolog-style logic resolution
* **Type Inference**: Hindley-Milner type unification
* **Constraint Solving**: Unification constraints in SMT

References
=====================

- Robinson, J. A. (1965). *A Machine-Oriented Logic Based on the Resolution Principle*. JACM 1965
- Martelli, A., & Montanari, U. (1982). *An Efficient Unification Algorithm*. TOPLAS 1982
- Baader, F., & Snyder, W. (2001). *Unification Theory*. Handbook of Automated Reasoning
