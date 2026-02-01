# Unsat core

Computation of unsatisfiable cores (UCS) and related enumeration for SMT/CNF formulas.

This package provides a uniform interface to several algorithms:

- **MARCO** (`marco.py`): Enumerate minimal unsatisfiable cores (MUSes) and maximal satisfying subsets (MSSes) via the Liffiton–Malik / Previti–Marques-Silva approach.
- **MSS** (`mss.py`): Enumerate maximal satisfying subsets using maximal resolution.
- **MUSX** (`musx.py`): Deletion-based extraction of a single minimal unsatisfiable subset (plain and partial CNF).
- **OptUx** (`optux.py`): Smallest MUS (SMUS) extraction and enumeration via implicit hitting set dualization; supports weighted WCNF.

Use `UnsatCoreComputer` in `unsat_core.py` with strategy `Algorithm.MARCO`, `Algorithm.MUSX`, or `Algorithm.OPTUX` to run the desired backend.
