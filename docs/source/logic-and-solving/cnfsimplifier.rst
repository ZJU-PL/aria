CNF Simplification Framework
=============================

``aria.bool.cnfsimplifier`` is a dedicated CNF simplification framework
that applies a battery of preprocessing and in-processing techniques to
reduce formula size before or during solving.

Directory structure
-------------------

::

   aria/bool/cnfsimplifier/
   ├── cnf.py              # Internal CNF representation
   ├── clause.py           # Clause data structure with literal operations
   ├── variable.py         # Variable activity and occurrence tracking
   ├── simplifier.py       # Main simplification engine (8 techniques)
   ├── io.py               # DIMACS / numeric-clause I/O helpers
   └── rust_backend.py     # Optional Rust-accelerated backend

Simplification techniques
--------------------------

The simplifier implements eight techniques:

1. **Tautology elimination** -- remove clauses containing both ``x`` and ``¬x``.
2. **Hidden tautology elimination (HTE)** -- detect and remove hidden tautologies.
3. **Asymmetric tautology elimination (ATE)** -- strengthen via resolution.
4. **Subsumption elimination** -- remove clauses subsumed by shorter ones.
5. **Hidden subsumption elimination** -- subsumption via resolution.
6. **Asymmetric subsumption elimination** -- asymmetric variant of subsumption.
7. **Blocked clause elimination (BCE)** -- remove clauses with blocking literals.
8. **Hidden blocked clause elimination** -- blocked-clause reasoning via resolution.

Key entry points
-----------------

* ``simplify_numeric_clauses(clauses)`` -- high-level API that takes a list of
  integer-list clauses (positive/negative literals) and returns the simplified
  clause set.
* Individual techniques are also available as standalone functions
  (``cnf_subsumption_elimination``, ``cnf_tautoly_elimination``, etc.).

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.cnfsimplifier import simplify_numeric_clauses

   clauses = [[1, 2], [1, -2, 3], [-1, 2, 3], [2, 3]]
   simplified = simplify_numeric_clauses(clauses)
   print(f"Before: {len(clauses)} clauses, After: {len(simplified)} clauses")
