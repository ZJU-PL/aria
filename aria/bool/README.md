# Boolean Reasoning Toolkit

A comprehensive collection of tools and algorithms for Boolean satisfiability (SAT), maximum satisfiability (MaxSAT), quantified Boolean formulas (QBF), and related logical reasoning tasks.

## Components

### Core SAT Solvers
- **SAT solvers**: PySAT, Z3, and brute force implementations
- **MaxSAT solvers**: Multiple algorithms including FM, LSU, RC2, Anytime
- **QBF solvers**: Support for QDIMACS and QCIR formats

### Formula Manipulation
- **CNF simplification**: Tautology elimination, subsumption, blocked clause removal
- **Tseitin transformation**: DNF to CNF conversion with auxiliary variables
- **NNF (Negation Normal Form)**: Full manipulation and reasoning capabilities

### Advanced Features
- **Dissolve**: Distributed SAT solver based on Stålmarck's method with dilemma splits
- **Feature extraction**: SATzilla-style features for SAT instance analysis
- **Knowledge compilation**: DNNF, OBDD compilation from logical formulas
- **Boolean interpolation**: Proof-based and core-based algorithms

### Usage

```python
# SAT solving
from aria.bool.sat.pysat_solver import PySATSolver
solver = PySATSolver()
result = solver.solve(cnf_formula)

# MaxSAT solving
from aria.bool.maxsat import MaxSATSolver
maxsat_solver = MaxSATSolver()
result = maxsat_solver.solve(weighted_cnf)

# CNF simplification
from aria.bool.cnf_simplify import parse_dimacs, write_dimacs
cnf = parse_dimacs("input.cnf")
simplified = cnf.tautology_elimination()
write_dimacs(simplified, "output.cnf")

# Tseitin transformation
from aria.bool.tseitin_converter import tseitin
cnf_result = tseitin(dnf_formula)
```

## Submodules

- `cnfsimplifier/`: Advanced CNF manipulation and simplification (optional Rust backend in `cnfsimplifier_rs/`)
- `dissolve/`: Distributed SAT solving with dilemma rules
- `features/`: SAT instance feature extraction and analysis
- `interpolant/`: Boolean interpolation algorithms
- `knowledge_compiler/`: Knowledge compilation to DNNF/OBDD
- `maxsat/`: Maximum satisfiability solvers
- `nnf/`: Negation normal form reasoning
- `qbf/`: Quantified Boolean formula support
- `sat/`: Core SAT solver implementations
