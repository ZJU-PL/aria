# Z3 Utilities

Simple descriptions of the Python files in `aria/utils/z3`:

- `__init__.py`: re-exports the Z3 helper modules in this package. Example APIs:
  package-level imports from `expr.py`, `solver.py`, `opt.py`, `bv.py`, `uf.py`,
  `values.py`, `ext.py`, and `cp.py`.
- `expr.py`: helpers for inspecting and transforming Z3 expressions and
  formulas. Example APIs: `get_variables`, `get_atoms`, `skolemize`,
  `big_and`, `negate`, `get_z3_logic`.
- `solver.py`: small solver-based predicates and model/DNF utilities. Example
  APIs: `is_sat`, `is_unsat`, `is_valid`, `is_entail`, `to_dnf`, `get_models`.
- `opt.py`: wrappers for Z3 optimization and MaxSMT APIs. Example APIs:
  `optimize`, `box_optimize`, `pareto_optimize`, `maxsmt`.
- `bv.py`: bit-vector helpers, including extension operations and signedness
  checks. Example APIs: `zero_extension`, `sign_extension`,
  `right_zero_extension`, `get_signedness`, `Signedness`.
- `values.py`: helpers for converting and manipulating Z3 values, especially BV
  and FP values. Example APIs: `bool_to_bit_vec`, `bv_log2`, `zext_or_trunc`,
  `ctlz`, `cttz`, `fp_mod`.
- `uf.py`: utilities for working with uninterpreted functions and related
  rewrites. Example APIs: `visitor`, `modify`, `replace_func_with_template`,
  `instiatiate_func_with_axioms`, `purify`.
- `ext.py`: extra or experimental helpers for quantifiers and boolean DNF
  conversion. Example APIs: `ground_quantifier`, `ground_quantifier_all`,
  `reconstruct_quantified_formula`, `to_dnf_boolean`.
- `cp.py`: constraint-programming-style helpers and decompositions for global
  constraints. Example APIs: `makeIntVar`, `makeIntVars`, `all_different`,
  `element`, `global_cardinality_count`, `cumulative`.
