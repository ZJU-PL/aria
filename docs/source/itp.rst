Interactive Theorem Proving (ITP)
===============================

Overview
--------

The ``aria.itp`` module provides a framework for interactive theorem proving with SMT solvers. It offers a Python interface for writing formal proofs, defining axioms, and verifying mathematical properties.

This module serves as a bridge between Python and various SMT (Satisfiability Modulo Theories) solvers like Z3, CVC5, and Vampire, providing:

- A kernel that maintains the logical integrity of proofs
- Various tactics for constructing proofs
- Support for datatypes and inductive definitions
- A library of theories (arithmetic, sets, sequences, etc.)
- Notation helpers for writing mathematical formulas
- Utilities for simplification and rewriting

Core Components
--------------

Solvers
~~~~~~~

A shim layer in ``smt.py`` that enables using different SMT solvers:

- Z3 (default)
- CVC5
- Vampire

You can select the solver by setting the ``KNUCKLE_SOLVER`` environment variable before import.

.. code-block:: python

   import os
   os.environ["KNUCKLE_SOLVER"] = "cvc5"  # or "vampire"
   
   from aria.itp import *
   import aria.itp.smt as smt

Kernel
~~~~~~

The core logical system in ``kernel.py`` that ensures the soundness of proofs:

- ``Proof`` - Represents a proven theorem
- ``prove()`` - Attempts to prove a theorem with the selected SMT solver
- ``axiom()`` - Introduces an axiom (use with caution)
- ``define()`` - Creates definitions with their corresponding axioms

.. code-block:: python

   # Prove a theorem
   x = smt.Real('x')
   theorem = smt.Implies(x > 0, x*x > 0)
   proof = prove(theorem)
   
   # Define a function
   square = define("square", [x], x * x)

Tactics
~~~~~~~

Higher-level proof strategies in ``tactics.py``:

- ``Lemma`` - For defining and proving lemmas
- ``Calc`` - For equational reasoning
- Various other tactics for common proof patterns

.. code-block:: python

   @Lemma
   def square_positive(x):
       premise = x > 0
       conclusion = square(x) > 0
       return smt.Implies(premise, conclusion)
   
   # Equational reasoning
   @Lemma
   def square_expand(a, b):
       calc = Calc(square(a + b))
       calc.equals((a + b) * (a + b))
       calc.equals(a*a + a*b + b*a + b*b)
       calc.equals(a*a + 2*a*b + b*b)
       return calc.proof

Datatypes
~~~~~~~~~

Support for defining and working with complex data structures in ``datatype.py``:

- ``Struct`` - For product types
- ``Enum`` - For sum types
- ``Inductive`` - For inductive datatypes
- ``InductiveRel`` - For inductive relations

.. code-block:: python

   # Define a list datatype
   List = Inductive("List")
   List.declare("nil")
   List.declare("cons", ("head", smt.IntSort()), ("tail", List))
   List = List.create()
   
   # Define a point struct
   Point = Struct("Point", x=smt.RealSort(), y=smt.RealSort())

Theories
~~~~~~~~

Mathematical theories implemented in the ``theories`` directory:

- Basic types: ``int``, ``bool``, ``nat``, ``bitvec``, ``float``
- Data structures: ``list``, ``set``, ``seq``, ``option``
- Algebraic structures in ``algebra/``
- Real analysis in ``real/``
- Logical foundations in ``logic/``

.. code-block:: python

   # Import a theory
   from aria.itp.theories import set
   
   # Use the theory
   A = smt.Const('A', set.SetSort(smt.IntSort()))
   B = smt.Const('B', set.SetSort(smt.IntSort()))
   
   # Prove a property about sets
   union_commutative = prove(set.union(A, B) == set.union(B, A))

Notation
~~~~~~~~

Helpers for writing mathematical formulas in ``notation.py``:

- ``QForAll`` - Universal quantifier with nicer syntax
- ``QExists`` - Existential quantifier with nicer syntax
- ``cond`` - Conditionals

.. code-block:: python

   # Define variables
   x = smt.Real('x')
   y = smt.Real('y')
   
   # Universal quantification
   forall_xy = QForAll([x, y], x + y == y + x)
   
   # Existential quantification
   exists_x = QExists([x], x*x == smt.RealVal(4))

Rewriting
~~~~~~~~~

Support for term rewriting in ``rewrite.py``:

- ``simp`` - Simplifies terms according to rewrite rules

.. code-block:: python

   # Define a term
   x = smt.Real('x')
   term = x + 0
   
   # Simplify it
   simplified = simp(term)  # x

Advanced Features
----------------

Proofs by Induction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prove a property about list length
   @Lemma
   def length_non_negative(xs):
       # The property to prove
       P = smt.Lambda([xs], length(xs) >= 0)
       
       # Base case: nil
       base_case = prove(P(List.nil))
       
       # Inductive case
       x = smt.Int('x')
       xs1 = smt.Const('xs1', List)
       inductive_case = prove(
           smt.Implies(P(xs1), P(List.cons(x, xs1)))
       )
       
       # Apply induction principle
       return prove(P(xs), by=[base_case, inductive_case])

Debugging Proofs
~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       proof = prove(complex_theorem, timeout=10000, dump=True)
   except Exception as e:
       print(f"Proof failed: {e}")

External Solver Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The module supports integrating with external provers through the ``solvers`` package:

- ``VampireSolver`` - Interface to the Vampire theorem prover
- Other solvers like Gappa, EggLog, and APRove

Configuration
------------

Global configuration options in ``config.py``:

- ``solver`` - The default solver to use
- ``admit_enabled`` - Whether to allow admitting theorems without proof
- ``timing`` - Enable/disable performance logging

Installation
-----------

The module is part of the aria package. External solvers may need separate installation.
See ``solvers/install.sh`` for installing supported external solvers.

API Reference
------------

Core Classes
~~~~~~~~~~~

Proof
^^^^^

.. code-block:: python

   class Proof:
       def __init__(self, thm: smt.BoolRef, reason: list[Any], admit: bool = False)

Represents a proven theorem.

**Properties:**

- ``thm``: The theorem that has been proven (a Z3 BoolRef)
- ``reason``: The reasons or premises used in the proof
- ``admit``: Whether the theorem was admitted without proof

**Methods:**

- ``__call__(*args)``: Instance a universally quantified theorem with specific terms

Core Functions
~~~~~~~~~~~~~

prove
^^^^^

.. code-block:: python

   def prove(thm: smt.BoolRef, by: Proof | Iterable[Proof] = [], 
             admit=False, timeout=1000, dump=False, solver=None) -> Proof

Attempts to prove a theorem using the specified SMT solver.

**Parameters:**

- ``thm``: The theorem to prove
- ``by``: Previously proven lemmas to use as premises
- ``admit``: If True, admit the theorem without proof
- ``timeout``: Solver timeout in milliseconds
- ``dump``: Whether to dump solver state (for debugging)
- ``solver``: Custom solver to use

**Returns:**

- A ``Proof`` object representing the proven theorem

axiom
^^^^^

.. code-block:: python

   def axiom(thm: smt.BoolRef, by=["axiom"]) -> Proof

Declares an axiom (use with caution).

**Parameters:**

- ``thm``: The axiom to declare
- ``by``: A description or reason for the axiom

**Returns:**

- A ``Proof`` object for the axiom

define
^^^^^^

.. code-block:: python

   def define(name: str, args: list[smt.ExprRef], body: smt.ExprRef, 
              lift_lambda=False) -> smt.FuncDeclRef

Defines a new function and creates its corresponding definition axiom.

**Parameters:**

- ``name``: The name of the function
- ``args``: The arguments of the function
- ``body``: The body/definition of the function
- ``lift_lambda``: Whether to lift lambda expressions

**Returns:**

- The function declaration 