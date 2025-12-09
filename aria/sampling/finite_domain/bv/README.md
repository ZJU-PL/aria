# Bit-Vector (QF_BV) Samplers

This directory contains sampling algorithms for bit-vector formulas. All samplers implement the `Sampler` interface for consistency.

## Available Samplers

### 1. `BitVectorSampler` - Basic Enumeration
Simple enumeration-based sampler using blocking clauses.

**Use when:**
- You need guaranteed distinct samples
- Formula has relatively few solutions
- Simple and reliable behavior is preferred

**Pros:**
- Simple and reliable
- Works with any SMT solver
- Guarantees distinct samples

**Cons:**
- Not uniform sampling
- Limited scalability
- Can be slow for large solution spaces

```python
from aria.sampling.finite_domain import BitVectorSampler
from aria.sampling.base import SamplingOptions
import z3

x = z3.BitVec('x', 8)
formula = z3.And(x > 10, x < 100)

sampler = BitVectorSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10))

for sample in result:
    print(sample)
```

### 2. `HashBasedBVSampler` - XOR-based Uniform Sampling
Approximate uniform sampling using random XOR (parity) constraints.

**Use when:**
- You need approximately uniform samples
- Testing requires diverse, representative inputs
- You want better coverage of the solution space

**Reference:** https://github.com/Z3Prover/z3/issues/4675#issuecomment-686880139

**Pros:**
- Better approximation of uniform distribution
- Good for testing purposes
- Works well for large solution spaces

**Cons:**
- Not guaranteed uniform (approximate)
- May take longer to find samples
- Can fail to find samples in constrained spaces

```python
from aria.sampling.finite_domain import HashBasedBVSampler
from aria.sampling.base import SamplingOptions
import z3

x, y = z3.BitVecs('x y', 32)
formula = z3.And(z3.ULT(x, 100), z3.ULT(y, x))

sampler = HashBasedBVSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10, random_seed=42))

for i, sample in enumerate(result):
    print(f"Sample {i+1}: {sample}")
```

### 3. `QuickBVSampler` - Optimization-Guided Sampling
Efficient sampling using optimization-guided search with bit-flipping mutations.

**Use when:**
- Fuzzing or testing applications
- You need diverse samples quickly
- Single-variable sampling is acceptable

**Reference:** Rafael Dutra et al., "Efficient Sampling of SAT Solutions for Testing", ICSE 2018
- Paper: https://dl.acm.org/doi/10.1145/3180155.3180248
- Original: https://github.com/RafaelTupynamba/quicksampler/

**Note:** Currently samples one target variable at a time. By default, uses the first bit-vector variable found.

**Pros:**
- Efficient for generating diverse samples
- Good for testing/fuzzing
- Uses optimization to guide sampling
- Fast sample generation

**Cons:**
- Not uniform sampling
- Currently samples only one variable at a time
- Generated samples are not validated (TODO)

```python
from aria.sampling.finite_domain import QuickBVSampler
from aria.sampling.base import SamplingOptions
import z3

x = z3.BitVec('x', 16)
y = z3.BitVec('y', 16)
formula = z3.And(x > 1000, y < 10000, z3.Or(x < 4000, x > 5000))

# Sample the variable 'x'
sampler = QuickBVSampler(target_var=x)
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10))

for i, sample in enumerate(result):
    print(f"Sample {i+1}: {sample}")
```

## Comparison

| Sampler | Uniformity | Speed | Multi-Var | Use Case |
|---------|-----------|-------|-----------|----------|
| `BitVectorSampler` | No | Slow | Yes | Simple enumeration, few solutions |
| `HashBasedBVSampler` | Approximate | Medium | Yes | Testing, uniform-like distribution |
| `QuickBVSampler` | No | Fast | Single* | Fuzzing, diverse samples |

\* QuickBVSampler currently focuses on one target variable

## Choosing a Sampler

1. **For testing with good coverage**: Use `HashBasedBVSampler` for approximate uniform distribution
2. **For simple enumeration**: Use `BitVectorSampler` when you need all distinct solutions
3. **For fuzzing**: Use `QuickBVSampler` for fast, diverse samples on a target variable
4. **For multiple variables**: Use `BitVectorSampler` or `HashBasedBVSampler` (both support multi-var)

## Implementation Details

All samplers follow the same interface:

```python
class YourSampler(Sampler):
    def supports_logic(self, logic: Logic) -> bool:
        """Check if this sampler supports the logic."""

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """Initialize with a formula."""

    def sample(self, options: SamplingOptions) -> SamplingResult:
        """Generate samples."""

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """Return supported sampling methods."""

    def get_supported_logics(self) -> Set[Logic]:
        """Return supported logics."""
```

## Future Work

- Implement true UniGen uniform sampler for Boolean/BV formulas
- Extend QuickBVSampler to support multiple variables simultaneously
- Add sample validation to QuickBVSampler
- Implement adaptive sampling that chooses strategy based on formula characteristics
