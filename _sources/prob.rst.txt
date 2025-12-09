Probabilistic Reasoning (aria/prob)
====================================

Overview
--------

The ``aria.prob`` package provides utilities for probabilistic reasoning over
logical formulas.

Currently supported:

- Weighted Model Counting (WMC) for propositional CNF formulas
  - Exact evaluation via DNNF compilation
  - SAT-based model enumeration backend (useful for sanity checks or small formulas)
- Weighted Model Integration (WMI): API stub for future LRA/LIA support

API
---

.. code-block:: python

    from pysat.formula import CNF
    from aria.prob import wmc_count, WMCBackend, WMCOptions

    cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
    weights = {1: 0.6, 2: 0.7, 3: 0.5}
    res = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))
    print(res)

Notes
-----

- Literal weights are probabilities in ``[0,1]``. If only one polarity is
  provided for a variable, the other is assumed to be ``1 - w``.
- The DNNF backend relies on the knowledge compilation utilities in
  ``aria.bool.knowledge_compiler``.
- The enumeration backend is intended for small formulas and debugging.
