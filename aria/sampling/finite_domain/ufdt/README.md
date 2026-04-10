# UF + Datatype (QF_UFDT) Samplers

This directory contains samplers for formulas that mix uninterpreted functions
and algebraic datatypes.

## Available Samplers

### `MixedUFDatatypeSampler`
Enumeration-based sampler over constant symbols and ground UF applications
appearing in the formula.

**Use when:**
- The formula mixes datatype-valued variables with UF applications
- You want projected uniqueness over a subset of constants or terms
- A finite observed term space is sufficient for sampling

**Notes:**
- Datatype constants are sampled just like other tracked constants.
- `projection_terms` controls both the returned sample keys and the uniqueness
  criterion used by blocking clauses.
- `return_full_model=True` returns the full tracked model while keeping
  uniqueness defined by `projection_terms`.
- `tracked_terms=[...]` returns a custom set of constants/terms independently of
  the uniqueness projection.
