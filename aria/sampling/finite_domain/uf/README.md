# Uninterpreted-Function (`QF_UF`, `QF_UFLIA`) Samplers

This directory contains samplers for quantifier-free formulas with
uninterpreted functions, including formulas that also contain linear integer
arithmetic.

## Available Samplers

### `UninterpretedFunctionSampler`
Enumeration-based sampler over the ground uninterpreted constants and function
applications that occur in the input formula.

**Use when:**
- You want concrete samples for observed UF terms
- The formula is ground or effectively ground after variable assignment
- Relevant domains are finite or finitely constrained by the formula
- You want projected samples for mixed UF + integer terms in `QF_UFLIA`

**Notes:**
- Samples are projections onto the constants and UF applications appearing in
  the formula.
- For `QF_UFLIA`, integer variables are also tracked so projected enumeration
  can block and report mixed UF/arithmetic assignments.
- The sampler blocks duplicate assignments over those tracked terms rather than
  attempting to enumerate whole function graphs.
- Use `SamplingOptions(num_samples=..., projection_terms=[...])` to project the
  sampled space onto selected constants or ground UF terms.
- Use `return_full_model=True` to keep projected uniqueness while returning all
  tracked UF constants and terms.
- Use `tracked_terms=[...]` to choose a custom output projection separately from
  the uniqueness projection.
