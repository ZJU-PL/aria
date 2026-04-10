# Finite Domain Samplers

This directory contains samplers for finite domain formulas, organized by SMT theory. All samplers implement a consistent interface via the `Sampler` base class.

## Directory Structure

```
finite_domain/
├── bool/              # Boolean (SAT) samplers
│   ├── base.py        # BooleanSampler - basic enumeration
│   └── README.md      # Documentation and usage examples
├── bv/                # Bit-vector (QF_BV) samplers
│   ├── base.py            # BitVectorSampler - basic enumeration
│   ├── hash_sampler.py    # HashBasedBVSampler - XOR-based uniform sampling
│   ├── quick_sampler.py   # QuickBVSampler - optimization-guided sampling
│   └── README.md          # Documentation and usage examples
├── uf/                # Uninterpreted-function (QF_UF) samplers
│   ├── base.py        # UninterpretedFunctionSampler - ground UF enumeration
│   └── README.md      # Documentation and usage examples
├── dt/                # Datatype (QF_DT) samplers
│   ├── base.py        # DatatypeSampler - datatype-valued enumeration
│   └── README.md      # Documentation and usage examples
├── ufdt/              # Mixed UF+datatype (QF_UFDT) samplers
│   ├── base.py        # MixedUFDatatypeSampler - composed UF/DT enumeration
│   └── README.md      # Documentation and usage examples
└── __init__.py        # Main exports
```

## Quick Start

All samplers follow a consistent 3-step pattern:

```python
from aria.sampling.finite_domain import SamplerClass
from aria.sampling.base import SamplingOptions

# 1. Create sampler
sampler = SamplerClass()

# 2. Initialize with formula
sampler.init_from_formula(your_formula)

# 3. Generate samples
result = sampler.sample(SamplingOptions(num_samples=10))
```

### Boolean Sampling Example
```python
from aria.sampling.finite_domain import BooleanSampler
from aria.sampling.base import SamplingOptions
import z3

# Create a Boolean formula
a, b, c = z3.Bools('a b c')
formula = z3.And(z3.Or(a, b), z3.Or(b, c))

# Sample models
sampler = BooleanSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=5))

for sample in result:
    print(sample)  # e.g., {'a': True, 'b': False, 'c': True}
```

### Bit-Vector Sampling Example
```python
from aria.sampling.finite_domain import HashBasedBVSampler
from aria.sampling.base import SamplingOptions
import z3

# Create a bit-vector formula
x, y = z3.BitVecs('x y', 32)
formula = z3.And(z3.ULT(x, 100), z3.ULT(y, x))

# Sample with uniform-like distribution
sampler = HashBasedBVSampler()
sampler.init_from_formula(formula)
result = sampler.sample(SamplingOptions(num_samples=10, random_seed=42))

for sample in result:
    print(sample)  # e.g., {'x': 45, 'y': 23}
```

## Available Samplers

### Boolean Samplers (`bool/`)

| Sampler | Strategy | Use Case |
|---------|----------|----------|
| `BooleanSampler` | Enumeration with blocking clauses | Simple, reliable sampling |

### Bit-Vector Samplers (`bv/`)

| Sampler | Strategy | Use Case |
|---------|----------|----------|
| `BitVectorSampler` | Enumeration with blocking clauses | Simple enumeration |
| `HashBasedBVSampler` | XOR-based hashing | Uniform-like sampling |
| `QuickBVSampler` | Optimization-guided mutations | Fast diverse samples |

See `bv/README.md` for detailed comparisons and usage guidance.

### UF Samplers (`uf/`)

| Sampler | Strategy | Use Case |
|---------|----------|----------|
| `UninterpretedFunctionSampler` | Enumeration over ground UF terms | Ground QF_UF formulas with finite observed term space |

### Datatype Samplers (`dt/`)

| Sampler | Strategy | Use Case |
|---------|----------|----------|
| `DatatypeSampler` | Enumeration with blocking clauses | Enumeration datatypes and finite ADTs |

### UF+Datatype Samplers (`ufdt/`)

| Sampler | Strategy | Use Case |
|---------|----------|----------|
| `MixedUFDatatypeSampler` | Enumeration over constants, datatype vars, and ground UF terms | Mixed `QF_UFDT` formulas |

## Common Options

All samplers accept `SamplingOptions`:

```python
from aria.sampling.base import SamplingOptions, SamplingMethod

options = SamplingOptions(
    num_samples=10,        # Number of samples to generate
    random_seed=42,        # For reproducibility
    timeout=30.0,          # Timeout in seconds (optional)
    method=SamplingMethod.ENUMERATION  # Sampling method
)
```

## Design Principles

### 1. **Consistent Interface**
All samplers implement the `Sampler` base class with uniform API:
- `init_from_formula()`: Initialize with a Z3 formula
- `sample()`: Generate samples with given options
- `supports_logic()`: Check if logic is supported
- `get_supported_methods()`: Query available methods

### 2. **Theory-Based Organization**
Samplers are grouped by SMT theory (Boolean, Bit-Vector) making it easy to:
- Find appropriate samplers for your logic
- Add new samplers for a specific theory
- Understand capabilities at a glance

### 3. **Class-Based API**
Only sampler classes are exported (not raw functions) to:
- Maintain consistent behavior across all samplers
- Enable state management and configuration
- Support polymorphism and factory patterns

### 4. **Separation of Strategy**
Different sampling strategies are separate classes, allowing users to:
- Choose the right algorithm for their needs
- Switch strategies without changing code structure
- Compare different approaches easily

## Choosing a Sampler

### By Logic
- **Boolean formulas (SAT)**: Use `BooleanSampler`
- **Bit-vector formulas (QF_BV)**: Choose from `BitVectorSampler`, `HashBasedBVSampler`, or `QuickBVSampler`
- **Uninterpreted functions (QF_UF)**: Use `UninterpretedFunctionSampler`
- **Algebraic datatypes (QF_DT)**: Use `DatatypeSampler`
- **UF + datatypes (QF_UFDT)**: Use `MixedUFDatatypeSampler`

## Projected Sampling

UF, DT, and UFDT samplers support projected enumeration via
`SamplingOptions(..., projection_terms=[...])`.

- `projection_terms` can contain Z3 expressions or their string names
- duplicate blocking is also performed only on the projected terms
- `return_full_model=True` keeps projected uniqueness but returns all tracked terms
- `tracked_terms=[...]` overrides the returned keys explicitly while preserving
  uniqueness from `projection_terms`

## Observable Term Model

The finite-domain samplers now distinguish between the full tracked observable
space and the default projection space.

- `QF_UF` tracks constants and ground UF applications appearing in the formula.
- `QF_DT` tracks datatype variables plus any datatype selectors/testers that
  syntactically appear in the formula.
- `QF_UFDT` combines both sets of observables.

For `QF_DT` and `QF_UFDT`, `SamplingOptions(..., include_selector_closure=True)`
adds a bounded one-level selector closure:

- if constructor evidence such as `box == some(x)` or `is_some(box)` appears,
  selector observations like `value(box)` become available
- constructor evidence is propagated across datatype equalities, so aliases such
  as `box == tag(x)` can expose `value(tag(x))`
- closure is intentionally bounded to one level to avoid term explosion

### By Requirements
- **Need uniform distribution**: Use `HashBasedBVSampler` (approximate uniform for BV)
- **Need all distinct solutions**: Use `BitVectorSampler` or `BooleanSampler`
- **Need fast diverse samples**: Use `QuickBVSampler` (for single BV variable)
- **Testing/fuzzing**: Use `HashBasedBVSampler` or `QuickBVSampler`

## Adding New Samplers

To add a new sampler:

1. Create a file in the appropriate theory directory (`bool/` or `bv/`)
2. Implement the `Sampler` interface:

```python
from aria.sampling.base import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult

class MyNewSampler(Sampler):
    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_BV  # or Logic.QF_BOOL

    def init_from_formula(self, formula):
        self.formula = formula
        # Extract variables, etc.

    def sample(self, options: SamplingOptions) -> SamplingResult:
        # Implement your sampling algorithm
        samples = [...]
        stats = {"time_ms": 0, "iterations": 0}
        return SamplingResult(samples, stats)

    def get_supported_methods(self):
        return {SamplingMethod.HASH_BASED}  # or appropriate method
```

3. Export in subdirectory's `__init__.py`
4. Update main `finite_domain/__init__.py`
5. Add documentation in subdirectory's README.md

## References

- **QuickSampler**: Rafael Dutra et al., "Efficient Sampling of SAT Solutions for Testing", ICSE 2018
- **XOR-based Sampling**: Chakraborty et al., "A Scalable Approximate Model Counter", CP 2013
- **UniGen**: Chakraborty et al., "Distribution-Aware Sampling and Weighted Model Counting for SAT", AAAI 2014

## Related Directories

- `../general_sampler/`: General-purpose samplers (MCMC, search tree, region-based)
- `../linear_ira/`: Samplers for linear integer/real arithmetic (QF_LIA, QF_LRA)
- `../nonlinear_ira/`: Samplers for non-linear arithmetic (QF_NRA, QF_NIA)
- `../base.py`: Base classes and interfaces for all samplers
