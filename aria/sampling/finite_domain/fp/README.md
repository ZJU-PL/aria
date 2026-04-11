# Floating-Point (QF_FP) Samplers

This directory contains finite-domain samplers for quantifier-free
floating-point formulas.

## Available Samplers

### `FloatingPointSampler` - IEEE-754 Enumeration

This sampler enumerates satisfying floating-point assignments with blocking
clauses over IEEE-754 bit patterns.

**Use when:**
- You need distinct satisfying assignments for `QF_FP` formulas
- Exact IEEE-754 values matter, including signed zeroes and NaNs
- A simple baseline sampler is sufficient

**Notes:**
- Blocking is done on `fpToIEEEBV(...)`, so distinct NaN payloads and signed
  zeroes are treated as different samples
- Returned sample values include both the simplified value and exact bits, e.g.
  `1.5 [bits=0x3fc00000]`

### `HashBasedFPSampler` - XOR over IEEE Encodings

This sampler adds random XOR constraints over the bits of `fpToIEEEBV(...)` to
obtain more diverse floating-point models.

**Use when:**
- You want broader `QF_FP` coverage than plain solver enumeration
- Approximate-uniform diversity is more important than deterministic ordering
- You are fuzzing or generating floating-point test inputs

**Notes:**
- Hashing works over IEEE-754 encodings, so the sampling space respects exact
  signed zeroes, infinities, and NaN payloads
- This is approximate and may return fewer than the requested number of samples
  for very constrained formulas

### `TotalOrderFPSampler` - IEEE Total-Order Spread

This sampler first enumerates a candidate pool, then selects a subset that is
spread out in IEEE `totalOrder` space.

**Use when:**
- You want deterministic coverage across different floating-point regions
- Approximate-uniform hashing is less important than deliberate numeric spread
- You want a sampler that distinguishes signed zeroes, infinities, and NaNs
  while avoiding clustered outputs

**Notes:**
- Selection is based on lexicographic tuples of IEEE `totalOrder` keys
- `candidate_pool_size` or `candidate_pool_factor` can be used to trade runtime
  for wider coverage

## Rendering Modes

All `QF_FP` samplers accept `SamplingOptions(..., render_mode=...)`:

- `pretty`: readable Z3-style floating-point values such as `1.5`
- `bits`: exact IEEE encoding only, such as `0x3fc00000`
- `pretty+bits`: readable value plus exact encoding, such as
  `1.5 [bits=0x3fc00000]`

The default is `pretty+bits`.

```python
from aria.sampling.base import SamplingMethod, SamplingOptions
from aria.sampling.finite_domain import (
    FloatingPointSampler,
    HashBasedFPSampler,
    TotalOrderFPSampler,
)
import z3

x = z3.FP("x", z3.Float32())
one = z3.FPVal(1.0, z3.Float32())
two = z3.FPVal(2.0, z3.Float32())
formula = z3.And(z3.fpGEQ(x, one), z3.fpLEQ(x, two))

sampler = FloatingPointSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=3))

for sample in result:
    print(sample)

hash_sampler = HashBasedFPSampler()
hash_sampler.init_from_formula(formula)
hash_result = hash_sampler.sample(
    SamplingOptions(num_samples=3, method=SamplingMethod.HASH_BASED, random_seed=7)
)

for sample in hash_result:
    print(sample)

spread_sampler = TotalOrderFPSampler()
spread_sampler.init_from_formula(formula)
spread_result = spread_sampler.sample(
    SamplingOptions(
        num_samples=3,
        method=SamplingMethod.TOTAL_ORDER,
        render_mode="bits",
        candidate_pool_factor=12,
    )
)

for sample in spread_result:
    print(sample)
```
