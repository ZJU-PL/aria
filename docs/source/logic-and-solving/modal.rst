Modal Logic
===========

``aria.bool.modal`` provides reasoning over propositional modal logic
formulas, including parsing, normalization, semantic evaluation, and
model checking / searching.

Directory structure
-------------------

::

   aria/bool/modal/
   ├── formula.py          # Formula AST: Atom, Not, And, Or, Implies, Box, Diamond
   ├── model.py            # Kripke model representation
   ├── semantics.py        # Semantic evaluation of modal formulas over Kripke models
   ├── normalization.py    # Negation Normal Form (NNF) conversion for modal formulas
   ├── parser.py           # Textual parser for modal logic expressions
   ├── search.py           # Model search / generation algorithms
   └── utils.py            # Utility helpers

Formula AST
------------

The ``formula`` module defines the abstract syntax tree for modal logic:

* ``Atom`` -- propositional variable
* ``Not``, ``And``, ``Or``, ``Implies`` -- Boolean connectives
* ``Box`` (□) -- necessity modality
* ``Diamond`` (◇) -- possibility modality

Kripke Models
--------------

``model.py`` provides the ``KripkeModel`` representation: a set of worlds,
a valuation mapping worlds to truth assignments, and an accessibility
relation between worlds.

Semantics
---------

``semantics.py`` evaluates modal formulas over Kripke models. It determines
whether a formula holds at a given world, respecting the standard Kripke
semantics for □ and ◇.

Normalization
-------------

``normalization.py`` converts modal formulas into Negation Normal Form (NNF),
pushing negations inward and eliminating implications.

Parser
------

``parser.py`` parses textual modal logic expressions (using ``~`` for
negation, ``[]`` for Box, ``<>`` for Diamond, ``->`` for implication)
into the formula AST.

Model Search
------------

``search.py`` implements algorithms for searching or constructing Kripke
models that satisfy given modal formulas.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.modal.parser import parse
   from aria.bool.modal.model import KripkeModel
   from aria.bool.modal.semantics import evaluate

   formula = parse("[](p -> <>q)")
   # Build or load a Kripke model, then evaluate
   result = evaluate(formula, model, world)
