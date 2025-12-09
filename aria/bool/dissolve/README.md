# Dissolve (Distributed SAT with Dilemma Rule)

This module implements a practical version of the paper “Dissolve: A Distributed SAT Solver based on Stålmarck’s Method” by Julien Henry, Aditya Thakur, Nicholas Kidd, and Thomas Reps
https://research.cs.wisc.edu/wpis/papers/tr1839.pdf.

It uses PySAT backends and Python multiprocessing.

## Features
- Asynchronous Dilemma-splits (Alg. 3)
- Clause sharing via a UBTree-like bucket store
- Variable picking with vote aggregation from learnts
- Budget strategies: constant or Luby sequence (per round)
- Distribution strategies: dilemma (2^k branches) or portfolio

## Usage
```python
from pysat.formula import CNF
from aria.bool.dissolve import Dissolve, DissolveConfig

cnf = CNF(from_clauses=[[1, 2], [-1, 3], [2, 3]])
cfg = DissolveConfig(k_split_vars=5, budget_strategy="luby", budget_unit=10000,
                     distribution_strategy="dilemma", max_rounds=50)
res = Dissolve(cfg).solve(cnf)
print(res)
```

Run the demo:
```bash
python -m aria.bool.dissolve.dissolve_demo
```
