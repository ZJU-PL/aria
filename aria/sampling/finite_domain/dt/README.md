# Datatype (QF_DT) Samplers

This directory contains samplers for quantifier-free algebraic datatype
formulas.

## Available Samplers

### `DatatypeSampler`
Enumeration-based sampler for datatype-valued variables using blocking clauses.

**Use when:**
- You work with enumeration sorts or finite algebraic datatypes
- You need distinct concrete datatype assignments
- You want a sampler with the same API as the existing Boolean/BV samplers

**Notes:**
- Constructor values are returned as strings for nullary constructors.
- Non-nullary constructor values are returned as structured dictionaries with
  `constructor` and `fields`.
- Use `projection_terms=[x, y]` to enumerate unique assignments only over the
  selected datatype variables.
- Use `return_full_model=True` to return all tracked datatype variables while
  still blocking only on the projected variables.
- Use `tracked_terms=[...]` to return a custom subset of datatype variables.
