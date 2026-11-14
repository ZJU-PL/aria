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
------------

``cnf.py`` computes a summary of a CNF formula (``CNFAnalysis``): variable and
clause counts, min/max/mean clause size, unit/binary/Horn clause counts,
positive/negative literal counts, and primal-graph edges and components.

Metrics
-------

``metrics.py`` provides general-purpose metrics over Boolean formulas and CNF
clause lists, complementing the feature extraction in ``aria.bool.features``
with lighter-weight structural measurements. It includes clause-length
histograms (``clause_length_histogram``), per-variable clause degrees
(``variable_clause_degrees``), literal balance, quantifier-prefix reports, and
a combined ``cnf_analysis_report`` helper.

QBF Analysis
------------

``qbf.py`` offers analysis utilities specific to Quantified Boolean Formulas,
such as quantifier depth evaluation, variable dependency analysis, and
related structural checks.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.analysis.cnf import analyze_cnf

   from pysat.formula import CNF
   metrics = analyze_cnf(CNF(from_clauses=[[1, 2], [-1, 3]]))
