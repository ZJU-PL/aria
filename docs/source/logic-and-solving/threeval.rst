Three-Valued Logic
==================

``aria.bool.threeval`` implements propositional reasoning in Kleene's
three-valued logic, where propositions can be ``true``, ``false``, or
``unknown``.

Directory structure
-------------------

::

   aria/bool/threeval/
   ├── propositional.py    # Three-valued formula AST and evaluation
   ├── parser.py           # Parser for three-valued logic expressions
   ├── adapters.py         # Conversion between three-valued formulas and Boolean NNF
   └── minimization.py     # Formula minimisation techniques

Three-valued semantics
----------------------

Formulas are evaluated under Kleene strong three-valued logic:

* ``true ∧ unknown = unknown``
* ``false ∧ unknown = false``
* ``true ∨ unknown = true``
* ``false ∨ unknown = unknown``

This is useful for reasoning about partial models, over-approximations, and
verification scenarios where some propositions are undetermined.

Strong Kleene is the default semantics; ``propositional.py`` also defines weak
Kleene, Łukasiewicz K3, and Gödel G3 semantics (see
``aria/bool/threeval/README.md``).

Key components
--------------

* **Propositional** (``propositional.py``) -- defines the three-valued
  formula AST and truth-table evaluation engine.
* **Parser** (``parser.py``) -- textual parser for three-valued logic
  expressions.
* **Adapters** (``adapters.py``) -- convert between three-valued formulas and
  Boolean NNF (``from_nnf`` / ``to_nnf``), with a configurable
  ``unknown_policy`` of ``error``, ``false``, or ``true``.
* **Minimisation** (``minimization.py``) -- semantic minimisation of
  three-valued formulas, including information-join formulas and OBDD-based
  helpers.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.threeval.propositional import Variable, And, TruthValue, evaluate

   x = Variable("x")
   y = Variable("y")
   formula = And(x, y)
   assignment = {"x": TruthValue.TRUE, "y": TruthValue.UNKNOWN}

   # Evaluate a three-valued formula over a partial assignment
   result = evaluate(formula, assignment)  # Returns TruthValue.UNKNOWN
