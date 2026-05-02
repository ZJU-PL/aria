Boolean Analysis
================

``aria.bool.analysis`` provides analytical tools for Boolean and QBF
formulas, including CNF structural metrics and QBF analysis utilities.

Directory structure
-------------------

::

   aria/bool/analysis/
   ├── cnf.py       # CNF structural analysis and metrics
   ├── metrics.py   # General Boolean formula metrics
   └── qbf.py       # QBF-specific analysis

CNF Analysis
-------------

``cnf.py`` computes structural metrics over CNF formulas, such as clause
length distributions, variable occurrence counts, and related statistics
used for understanding formula structure and hardness.

Metrics
--------

``metrics.py`` provides general-purpose metrics over Boolean formulas,
complementing the feature extraction in ``aria.bool.features`` with lighter-weight
structural measurements.

QBF Analysis
-------------

``qbf.py`` offers analysis utilities specific to Quantified Boolean Formulas,
such as quantifier depth evaluation, variable dependency analysis, and
related structural checks.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.analysis.cnf import analyse_cnf

   metrics = analyse_cnf(cnf_clauses)
