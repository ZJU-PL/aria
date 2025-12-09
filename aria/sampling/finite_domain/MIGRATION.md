# Migration Guide: Consistent Sampler Interface

This document explains the changes made to create a consistent interface for all samplers.

## What Changed?

### Before (Inconsistent Interface)
```python
from aria.sampling.finite_domain import (
    BooleanSampler,           # Class
    BitVectorSampler,         # Class
    get_uniform_samples_with_xor,  # Raw function!
    bvsampler                 # Raw function generator!
)

# Mixed usage patterns
sampler1 = BooleanSampler()
sampler1.init_from_formula(formula)
result1 = sampler1.sample(options)

# vs.
samples2 = get_uniform_samples_with_xor([x, y], formula, 10)  # Different API!

# vs.
generator3 = bvsampler(formula, x)  # Yet another API!
for value in generator3:
    ...
```

### After (Consistent Interface)
```python
from aria.sampling.finite_domain import (
    BooleanSampler,           # Class
    BitVectorSampler,         # Class
    HashBasedBVSampler,       # Class (was get_uniform_samples_with_xor)
    QuickBVSampler            # Class (was bvsampler)
)

# Uniform usage pattern for ALL samplers
sampler1 = BooleanSampler()
sampler1.init_from_formula(formula)
result1 = sampler1.sample(options)

sampler2 = HashBasedBVSampler()
sampler2.init_from_formula(formula)
result2 = sampler2.sample(options)

sampler3 = QuickBVSampler(target_var=x)
sampler3.init_from_formula(formula)
result3 = sampler3.sample(options)
```

## Migration Examples

### Example 1: XOR-based Sampling

**Before:**
```python
from aria.sampling.finite_domain import get_uniform_samples_with_xor
import z3

x, y = z3.BitVecs('x y', 32)
formula = z3.And(z3.ULT(x, 100), z3.ULT(y, x))

# Raw function call
samples = get_uniform_samples_with_xor([x, y], formula, 10)
for sample in samples:
    print(f"x={sample[0]}, y={sample[1]}")
```

**After:**
```python
from aria.sampling.finite_domain import HashBasedBVSampler
from aria.sampling.base import SamplingOptions
import z3

x, y = z3.BitVecs('x y', 32)
formula = z3.And(z3.ULT(x, 100), z3.ULT(y, x))

# Consistent class-based API
sampler = HashBasedBVSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10, random_seed=42))

for sample in result:
    print(sample)  # {'x': 45, 'y': 23}
```

### Example 2: QuickSampler

**Before:**
```python
from aria.sampling.finite_domain import bvsampler
import z3

x = z3.BitVec('x', 16)
formula = z3.And(x > 1000, x < 10000)

# Generator function
generator = bvsampler(formula, x)
samples = []
for i, value in enumerate(generator):
    if i >= 10:
        break
    samples.append(value)
```

**After:**
```python
from aria.sampling.finite_domain import QuickBVSampler
from aria.sampling.base import SamplingOptions
import z3

x = z3.BitVec('x', 16)
formula = z3.And(x > 1000, x < 10000)

# Consistent class-based API
sampler = QuickBVSampler(target_var=x)
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10))

for sample in result:
    print(sample)  # {'x': 5432}
```

## Benefits of the New Design

1. **Consistency**: All samplers use the same API pattern
2. **Discoverability**: All samplers are classes with clear names
3. **Configurability**: Easy to set options via `SamplingOptions`
4. **Polymorphism**: Can use any sampler interchangeably
5. **Factory Pattern**: Easy to create sampler factories
6. **Testing**: Easier to mock and test

## Common Pattern

All samplers now follow this 3-step pattern:

```python
# 1. Create sampler instance
sampler = SamplerClass(optional_args)

# 2. Initialize with formula
sampler.init_from_formula(your_formula)

# 3. Generate samples
result = sampler.sample(SamplingOptions(
    num_samples=10,
    random_seed=42,
    timeout=30.0
))

# Access samples
for sample in result:
    print(sample)  # Dict[str, Any]

# Access statistics
print(result.stats)
```

## Internal Functions

The old raw functions still exist but are now **internal** (prefixed with `_`):
- `get_uniform_samples_with_xor` → `_get_uniform_samples_with_xor`
- `bvsampler` → `_bvsampler`

These are implementation details and should not be used directly.

## Choosing the Right Sampler

| Need | Sampler | Notes |
|------|---------|-------|
| Boolean formulas | `BooleanSampler` | Basic enumeration |
| Bit-vector enumeration | `BitVectorSampler` | Simple blocking clauses |
| Uniform-like BV samples | `HashBasedBVSampler` | Approximate uniform |
| Fast diverse BV samples | `QuickBVSampler` | For testing/fuzzing |

See the README files for detailed comparisons and usage examples.

## Questions?

- See `README.md` for overview and quick start
- See `bool/README.md` for Boolean sampler details
- See `bv/README.md` for Bit-vector sampler details and comparisons
