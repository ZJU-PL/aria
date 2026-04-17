# Solution Sampling for Various Constraints

This module provides various sampling algorithms for different SMT theories. The goal is to generate diverse,
representative models (solutions) from the solution space of SMT formulas.

## Overview

The sampling module offers a unified interface for sampling models from SMT formulas across various logics. It provides:

- A common interface for different sampling methods
- Support for multiple SMT theories
- Easy-to-use factory pattern for creating samplers
- Consistent API across all sampler implementations

## Dependencies

### Required

- Z3 SMT Solver (core dependency)

```bash
pip install z3-solver
```

### Optional

- numpy: Required for continuous domain sampling (Dikin walk, etc.)
- pyeda: Required for BDD-based sampling methods
- pysmt: Enables using multiple SMT solvers as backends

## Supported Logics

- **QF_BOOL**: Quantifier-free Boolean logic
- **QF_BV**: Quantifier-free bit-vector logic
- **QF_UF**: Quantifier-free uninterpreted functions
- **QF_UFLIA**: Quantifier-free uninterpreted functions with linear integer arithmetic
- **QF_DT**: Quantifier-free algebraic datatypes
- **QF_UFDT**: Quantifier-free uninterpreted functions with datatypes
- **QF_DTLIA**: Quantifier-free algebraic datatypes with linear integer arithmetic
- **QF_LRA**: Quantifier-free linear real arithmetic
- **QF_LIA**: Quantifier-free linear integer arithmetic
- **QF_NRA**: Quantifier-free non-linear real arithmetic
- **QF_NIA**: Quantifier-free non-linear integer arithmetic
- **QF_LIRA**: Quantifier-free linear integer and real arithmetic

## Sampling Methods

- **ENUMERATION**: Simple enumeration of models
- **MCMC**: Markov Chain Monte Carlo
- **REGION**: Region-based sampling
- **SEARCH_TREE**: Search tree-based sampling
- **HASH_BASED**: Hash-based sampling
- **DIKIN_WALK**: Dikin walk for continuous domains

## Usage

### Basic Usage

```python
from aria.sampling import sample_models_from_formula, Logic, SamplingOptions, SamplingMethod
import z3

# Create a formula
x, y = z3.Reals('x y')
formula = z3.And(x + y > 0, x - y < 1)

# Sample models from the formula
options = SamplingOptions(
    method=SamplingMethod.ENUMERATION,
    num_samples=10
)
result = sample_models_from_formula(formula, Logic.QF_LRA, options)

# Print the models
for i, model in enumerate(result):
    print(f"Model {i+1}: {model}")
```

### Using Different Logics

```python
# Boolean logic
a, b, c = z3.Bools('a b c')
bool_formula = z3.And(z3.Or(a, b), z3.Or(b, c))
bool_result = sample_models_from_formula(bool_formula, Logic.QF_BOOL, options)

# Bit-vector logic
x = z3.BitVec('x', 8)
y = z3.BitVec('y', 8)
bv_formula = z3.And(x + y > 10, x * 2 < 100)
bv_result = sample_models_from_formula(bv_formula, Logic.QF_BV, options)

# Linear integer arithmetic
i, j = z3.Ints('i j')
lia_formula = z3.And(i + j > 0, i - j < 10)
lia_result = sample_models_from_formula(lia_formula, Logic.QF_LIA, options)

# Uninterpreted functions + linear integer arithmetic
u = z3.Int('u')
v = z3.Int('v')
f = z3.Function('f', z3.IntSort(), z3.IntSort())
uflia_formula = z3.And(u >= 0, u <= 1, v == f(u), v <= 2)
uflia_result = sample_models_from_formula(
    uflia_formula,
    Logic.QF_UFLIA,
    SamplingOptions(num_samples=10, projection_terms=[u, v, f(u)]),
)

# Datatypes + linear integer arithmetic
maybe = z3.Datatype('MaybeInt')
maybe.declare('none')
maybe.declare('some', ('value', z3.IntSort()))
maybe = maybe.create()
x = z3.Int('x')
box = z3.Const('box', maybe)
dtlia_formula = z3.And(x >= 0, x <= 2, box == maybe.some(x))
dtlia_result = sample_models_from_formula(
    dtlia_formula,
    Logic.QF_DTLIA,
    SamplingOptions(num_samples=10, include_selector_closure=True),
)

# Shape-first DTLIA sampling with diversity-aware selection
shape_result = sample_models_from_formula(
    dtlia_formula,
    Logic.QF_DTLIA,
    SamplingOptions(
        method=SamplingMethod.SEARCH_TREE,
        num_samples=3,
        include_selector_closure=True,
        max_shapes=4,
        candidates_per_shape=4,
        diversity_mode="max_distance",
    ),
)

# Coverage-guided DTLIA sampling for test generation
coverage_result = sample_models_from_formula(
    dtlia_formula,
    Logic.QF_DTLIA,
    SamplingOptions(
        method=SamplingMethod.SEARCH_TREE,
        num_samples=3,
        include_selector_closure=True,
        max_shapes=4,
        candidates_per_shape=6,
        diversity_mode="coverage_guided",
    ),
)
```

`QF_DTLIA` now supports two practical modes via `aria.sampling.dtlia`:

- `ENUMERATION`: direct projected model enumeration over integer variables and
  datatype observables
- `SEARCH_TREE`: shape-first sampling that
  - enumerates ADT constructor shapes first
  - lowers each shape to a residual arithmetic problem
  - samples integer payloads per shape
  - can use `diversity_mode="max_distance"` for mixed-theory spread
  - can use `diversity_mode="coverage_guided"` to prioritize constructor,
    alias, and integer-boundary coverage for testing workloads

`QF_NRA` and `QF_NIA` support two basic modes via
`aria.sampling.nonlinear_ira`:

- `ENUMERATION`: iterative model enumeration with blocking clauses over the
  extracted numeric variables
- `SEARCH_TREE`: approximate search-tree sampling over the extracted numeric
  variables

### Advanced Usage

```python
from aria.sampling import create_sampler

# Create a sampler directly
sampler = create_sampler(Logic.QF_LRA)
sampler.init_from_formula(formula)

# Sample with custom options
options = SamplingOptions(
    method=SamplingMethod.ENUMERATION,
    num_samples=5,
    timeout=10.0,
    random_seed=42,
    # Additional method-specific options
    use_blocking_clauses=True
)

result = sampler.sample(options)

# Access statistics
if result.stats:
    for key, value in result.stats.items():
        print(f"{key}: {value}")
```

### Extending with Custom Samplers

To implement a custom sampler, create a class that inherits from `Sampler` and implements the required methods:

```python
from aria.sampling import Sampler, Logic, SamplingMethod, SamplingOptions, SamplingResult

class MyCustomSampler(Sampler):
    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_LRA
    
    def init_from_formula(self, formula):
        self.formula = formula
    
    def sample(self, options: SamplingOptions) -> SamplingResult:
        # Implement your sampling algorithm here
        samples = [...]
        stats = {"time": 0.1, "iterations": 10}
        return SamplingResult(samples, stats)
    
    def get_supported_methods(self):
        return {SamplingMethod.ENUMERATION}

# Register your sampler
from aria.sampling import SamplerFactory
SamplerFactory.register(Logic.QF_LRA, MyCustomSampler)
```

## Architecture

The sampling module is designed with extensibility in mind:

- `Sampler`: Abstract base class defining the interface for all sampler implementations
- `SamplingOptions`: Class for configuring sampling options
- `SamplingResult`: Class for representing sampling results
- `SamplerFactory`: Factory for creating sampler instances
- `sample_models_from_formula()`: High-level API for sampling models from formulas

## References

1. Dutra et al., "Efficient Sampling of SAT Solutions for Testing", ICSE 2018
2. Ermon et al., "Uniform Solution Sampling Using a Constraint Solver As an Oracle", UAI 2012
3. Chakraborty et al., "A Scalable Approximate Model Counter", CP 2013
