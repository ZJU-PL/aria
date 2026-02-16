# Prob Modelabilistic Reasoning

Weighted Counting (WMC) and Weighted Model Integration (WMI).

## Components

- `wmc/`: Main probabilistic reasoning module

## Weighted Model Counting

Compute weighted sum of satisfying assignments for propositional formulas.

Features:
- DNNF-based exact evaluation
- SAT-based enumeration backend
- Support for weighted CNF formulas

## Weighted Model Integration

Extend WMC to continuous domains with density functions.

Features:
- Monte Carlo integration
- Region-based integration
- Support for common distributions:
  - Uniform
  - Gaussian
  - Exponential
  - Beta

## Usage

```python
# Weighted Model Counting
from pysat.formula import CNF
from aria.prob.wmc import wmc_count, WMCBackend, WMCOptions

cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
weights = {1: 0.6, -1: 0.4, 2: 0.7, -2: 0.3, 3: 0.5, -3: 0.5}
result = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))

# Weighted Model Integration
import z3
from aria.prob.wmc import wmi_integrate, WMIOptions, UniformDensity

x, y = z3.Reals('x y')
formula = z3.And(x + y > 0, x < 1, y < 1, x > 0, y > 0)
density = UniformDensity({'x': (0, 1), 'y': (0, 1)})
result = wmi_integrate(formula, density, WMIOptions(num_samples=10000))
```

See `wmc/__init__.py` for detailed API.
