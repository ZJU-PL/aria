# Uninterpreted-Function (QF_UF) Samplers

This directory contains samplers for quantifier-free formulas with
uninterpreted functions.

## Available Samplers

### `UninterpretedFunctionSampler`
Enumeration-based sampler over the ground uninterpreted constants and function
applications that occur in the input formula.

**Use when:**
- You want concrete samples for observed UF terms
- The formula is ground or effectively ground after variable assignment
- Relevant domains are finite or finitely constrained by the formula

**Notes:**
- Samples are projections onto the constants and UF applications appearing in
  the formula.
- The sampler blocks duplicate assignments over those tracked terms rather than
  attempting to enumerate whole function graphs.
- Use `SamplingOptions(num_samples=..., projection_terms=[...])` to project the
  sampled space onto selected constants or ground UF terms.
- Use `return_full_model=True` to keep projected uniqueness while returning all
  tracked UF constants and terms.
- Use `tracked_terms=[...]` to choose a custom output projection separately from
  the uniqueness projection.
