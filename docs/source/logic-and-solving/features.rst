SAT Instance Feature Extraction
===============================

``aria.bool.features`` implements SATzilla-style feature extraction for CNF
formulas. It computes 40+ structural, statistical, and probing-based
features that characterise SAT instances, useful for algorithm selection and
machine-learning applications.

Directory structure
-------------------

::

   aria/bool/features/
   ├── sat_instance.py      # High-level API: SATInstance class
   ├── base_features.py     # Core feature extraction (size, graph, balance)
   ├── active_features.py   # Active feature computation
   ├── balance_features.py  # Positive/negative literal balance statistics
   ├── graph_features.py    # Variable-clause and variable-variable graph features
   ├── dpll.py              # DPLL probing (unit propagation at various depths)
   ├── parse_cnf.py         # CNF file parser
   ├── enums.py             # Feature name enumerations
   ├── array_stats.py       # Statistical helpers (mean, std, entropy, …)
   └── stopwatch.py         # Timing utility

Key class
---------

* ``SATInstance`` -- main entry point; loads a CNF file and exposes feature
  extraction via its attributes and helper methods.

Feature categories
------------------

* **Size features** -- number of clauses, variables, clause-to-variable ratio.
* **Variable-clause graph** -- degree statistics (mean, coefficient of
  variation, min, max) and entropy.
* **Variable graph** -- co-occurrence graph statistics.
* **Balance features** -- positive/negative literal ratios per variable and
  clause.
* **Horn clause proximity** -- fraction of Horn clauses and per-variable
  Horn-clause counts.
* **DPLL probing** -- unit propagation statistics at decision depths 1, 4, 16,
  64, and 256 (``dpll.py``).

The upstream SATzilla feature list in ``aria/bool/features/README.md`` also
mentions local-search probing (SAPS/GSAT), power-law/modularity/fractal
structure features, and eight weighted graphs. Those are not implemented in the
current Python package: ``gen_local_search_probing_features()`` raises
``NotImplementedError``, and ``base_features.py`` only computes the basic
SATzilla features (1--33).

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.features.sat_instance import SATInstance

   instance = SATInstance([[1, 2], [-1, 3], [-2, 3]])
   instance.gen_basic_features()
   # Access extracted features as a dictionary
   features = instance.features_dict
   print(features["v"], features["c"])  # variables, clauses
