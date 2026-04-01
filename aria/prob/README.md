# Probabilistic Reasoning

`aria.prob` now exposes a small but usable probabilistic inference layer:

- exact weighted model counting for Boolean CNF formulas
- explicit arithmetic probability-mass estimation backends
- high-level probability / conditional-probability / expectation / variance queries

Internally, the package is split into:

- `aria.prob.core` for shared densities, result types, and helper utilities
- `aria.prob.boolean` for Boolean WMC
- `aria.prob.arithmetic` for arithmetic WMI
- `aria.prob.api` for high-level query helpers

## Public API

```python
from aria.prob import (
    WMCOptions,
    WMIOptions,
    WMIMethod,
    DiscreteFactorizedDensity,
    UniformDensity,
    compile_wmc,
    conditional_probability,
    discrete_density,
    expectation,
    probability,
    variance,
    wmc_count,
    wmi_integrate,
)
```

## Boolean WMC

Use `wmc_count()` for a one-shot exact weighted count:

```python
from pysat.formula import CNF
from aria.prob import wmc_count

cnf = CNF(from_clauses=[[1, 2]])
weights = {1: 0.2, -1: 0.8, 2: 0.3, -2: 0.7}
count = wmc_count(cnf, weights)
```

Use `compile_wmc()` for repeated evidence queries:

```python
compiled = compile_wmc(cnf, weights)
px_given_model = compiled.probability(query=[1])
py_given_not_x = compiled.probability(query=[2], evidence=[-1])
```

Compiled objects also support repeated CNF conditional queries:

```python
query_cnf = CNF(from_clauses=[[2]])
evidence_cnf = CNF(from_clauses=[[1]])
result = compiled.probability_cnf(query_cnf, evidence_cnf)
```

`probability(cnf, weights, evidence=...)` is also available for one-shot CNF queries.
For the high-level CNF query API, complementary literal weights must sum to `1.0`.

## Arithmetic Probability Queries

`wmi_integrate()` returns an `InferenceResult`, which behaves like a float but
also exposes:

- `value`
- `exact`
- `backend`
- `stats`
- `error_bound`

For sampling-based moment queries, `stats` now includes:

- `conditioning_mass_estimate` / `conditioning_probability_estimate`
- `conditioning_mass_confidence_half_width`
- `moment_confidence_half_widths`
- per-draw weighted accumulator summaries for debugging
- `approx_effective_sample_size` for importance sampling

Example with bounded-support Monte Carlo:

```python
import z3
from aria.prob import UniformDensity, WMIOptions, WMIMethod, wmi_integrate

x, y = z3.Reals("x y")
triangle = z3.And(x >= 0, y >= 0, x <= 1, y <= 1, x + y <= 1)
density = UniformDensity({"x": (0, 1), "y": (0, 1)})

result = wmi_integrate(
    triangle,
    density,
    WMIOptions(
        method=WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO,
        num_samples=10000,
        random_seed=7,
    ),
)
```

High-level probability, expectation, and variance helpers:

```python
mass = probability(triangle, density)
cond = conditional_probability(triangle, x <= 0.5, density)
ex = expectation(x, triangle, density)
var = variance(x, triangle, density)
```

For `moment()`, `expectation()`, `covariance()`, and `variance()`, the returned
`error_bound` is an approximate normal-interval half-width for the reported
moment estimate. It is a diagnostic, not a rigorous finite-sample guarantee.
Likewise, `effective_conditioning_weight` refers to the estimated probability
mass of the conditioning event for sampling-based queries, and to the exact
conditioning probability for exact discrete queries.

## Exact Discrete Hook

For bounded integer formulas, `UniformDensity(..., discrete=True)` enables an
exact discrete-uniform path:

```python
x = z3.Int("x")
density = UniformDensity({"x": (0, 2)}, discrete=True)
mass = wmi_integrate(x < 2, density)
```

This path currently supports:

- Int variables only
- uniform discrete density over finite integer boxes
- formulas accepted by the arithmetic counting utilities

For non-uniform but factorized integer priors, use
`DiscreteFactorizedDensity` or `discrete_density(...)`:

```python
x = z3.Int("x")
density = discrete_density({"x": {0: 0.1, 1: 0.3, 2: 0.6}})
mass = wmi_integrate(x >= 1, density)
```

## Supported / Unsupported

Supported in this iteration:

- exact Boolean CNF WMC
- repeated literal evidence queries on compiled Boolean models
- repeated CNF evidence/query reuse on compiled Boolean models
- bounded-support Monte Carlo for finite rectangular supports
- importance sampling when the density or proposal can generate samples
- exact discrete uniform and factorized finite probability mass for bounded integer formulas

Documented arithmetic backends:

- `auto`
- `bounded_support_monte_carlo`
- `importance_sampling`
- `exact_discrete`

Explicitly unsupported or rejected:

- silent fallback from correlated Gaussian covariance to diagonal
- exact unbounded continuous integration
- nonlinear exact WMI
- ambiguous high-level CNF probabilities with non-normalized complementary weights
