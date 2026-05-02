Three-Valued Logic
===================

``aria.bool.threeval`` implements propositional reasoning in Kleene's
three-valued logic, where propositions can be ``true``, ``false``, or
``unknown``.

Directory structure
-------------------

::

   aria/bool/threeval/
   ├── propositional.py    # Three-valued formula AST and evaluation
   ├── parser.py           # Parser for three-valued logic expressions
   ├── adapters.py         # Adapters bridging three-valued and two-valued worlds
   └── minimization.py     # Formula minimisation techniques

Three-valued semantics
-----------------------

Formulas are evaluated under Kleene strong three-valued logic:

* ``true ∧ unknown = unknown``
* ``false ∧ unknown = false``
* ``true ∨ unknown = true``
* ``false ∨ unknown = unknown``

This is useful for reasoning about partial models, over-approximations, and
verification scenarios where some propositions are undetermined.

Key components
--------------

* **Propositional** (``propositional.py``) -- defines the three-valued
  formula AST and truth-table evaluation engine.
* **Parser** (``parser.py``) -- textual parser for three-valued logic
  expressions.
* **Adapters** (``adapters.py``) -- bridge utilities that translate between
  three-valued assignments and classical two-valued SAT/QBF encodings.
* **Minimisation** (``minimization.py``) -- algorithms for simplifying or
  minimising three-valued formulas.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.threeval.propositional import evaluate

   # Evaluate a three-valued formula over a partial assignment
   result = evaluate(formula, assignment)
