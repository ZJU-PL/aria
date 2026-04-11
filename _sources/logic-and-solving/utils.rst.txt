Utilities
=========

``aria.utils`` contains shared helper code used across ARIA. The package mixes
general utilities, solver-facing helpers, and Z3-specific helpers.

.. contents:: Table of Contents
   :local:
   :depth: 2

Package Layout
--------------

Important areas inside ``aria.utils``:

- ``types.py``: shared enums and solver-related types.
- ``exceptions.py``: common ARIA exception types.
- ``sexpr.py``: S-expression parsing helpers.
- ``solver/``: solver-process and SMT-LIB helpers.
- ``z3/``: Z3-specific expression, solver, optimization, and value helpers.

Z3 Utilities
------------

The ``aria.utils.z3`` package collects the Z3-focused helper modules.

- ``__init__.py``: re-exports the Z3 helper modules in this package. Example
  APIs: package-level imports from ``expr.py``, ``solver.py``, ``opt.py``,
  ``bv.py``, ``uf.py``, ``values.py``, ``ext.py``, and ``cp.py``.
- ``expr.py``: helpers for inspecting and transforming Z3 expressions and
  formulas. Example APIs: ``get_variables``, ``get_atoms``, ``skolemize``,
  ``big_and``, ``negate``, ``get_z3_logic``.
- ``solver.py``: small solver-based predicates and model/DNF utilities.
  Example APIs: ``is_sat``, ``is_unsat``, ``is_valid``, ``is_entail``,
  ``to_dnf``, ``get_models``.
- ``opt.py``: wrappers for Z3 optimization and MaxSMT APIs. Example APIs:
  ``optimize``, ``box_optimize``, ``pareto_optimize``, ``maxsmt``.
- ``bv.py``: bit-vector helpers, including extension operations and signedness
  checks. Example APIs: ``zero_extension``, ``sign_extension``,
  ``right_zero_extension``, ``get_signedness``, ``Signedness``.
- ``values.py``: helpers for converting and manipulating Z3 values, especially
  bit-vector and floating-point values. Example APIs: ``bool_to_bit_vec``,
  ``bv_log2``, ``zext_or_trunc``, ``ctlz``, ``cttz``, ``fp_mod``.
- ``uf.py``: utilities for working with uninterpreted functions and related
  rewrites. Example APIs: ``visitor``, ``modify``,
  ``replace_func_with_template``, ``instiatiate_func_with_axioms``, ``purify``.
- ``ext.py``: extra or experimental helpers for quantifiers and boolean DNF
  conversion. Example APIs: ``ground_quantifier``,
  ``ground_quantifier_all``, ``reconstruct_quantified_formula``,
  ``to_dnf_boolean``.
- ``cp.py``: constraint-programming-style helpers and decompositions for global
  constraints. Example APIs: ``makeIntVar``, ``makeIntVars``,
  ``all_different``, ``element``, ``global_cardinality_count``, ``cumulative``.

Other Core Utilities
--------------------

- ``types.py``: shared enums such as solver result and platform identifiers.
- ``exceptions.py``: base and shared exception types used across ARIA.
- ``sexpr.py``: parser utilities for SMT-LIB-style S-expressions.
- ``solver/smtlib.py``: SMT-LIB process helpers for external solvers.
- ``solver/pysmt.py``: PySMT-backed helpers.
- ``solver/pysat.py``: PySAT-backed helpers.
- ``solver/z3plus.py``: external-solver workflows around Z3-based pipelines.

Import Guidance
---------------

- Use ``aria.utils.z3.*`` for Z3-specific helpers.
- Use ``aria.utils.solver.*`` for external solver orchestration and SMT-LIB
  process handling.
- Use top-level modules such as ``aria.utils.types`` and
  ``aria.utils.exceptions`` for shared infrastructure.

Notes
-----

- ``aria.utils`` is a shared dependency across many ARIA subsystems.
- Prefer importing from the narrowest submodule that provides the API you need.
- In ``aria.utils.z3``, ``expr.py``, ``solver.py``, ``opt.py``, ``bv.py``,
  ``uf.py``, and ``values.py`` form the main core; ``ext.py`` is more
  experimental, and ``cp.py`` is a convenience layer for CP-style encodings.
