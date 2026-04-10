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
- **Boolean backbone**: multiple SAT-level algorithms for implied literal extraction
- **Modal logic**: finite Kripke semantics, parsing, normalization, model utilities, bounded witness search
- **Prime implicants / implicates**: SAT-based enumeration of minimal terms and clauses

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

# Prime implicants / implicates
from aria.bool.prime import enumerate_prime_implicants, enumerate_prime_implicates
prime_implicants = enumerate_prime_implicants(CNF(from_clauses=[[1, 2], [-1, 3]]))
prime_implicates = enumerate_prime_implicates(CNF(from_clauses=[[1, 2], [-1, 3]]))

# Backbone literals
from aria.bool.backbone import compute_backbone
backbone, calls = compute_backbone(CNF(from_clauses=[[1, 2], [-1, 3], [-2, 3]]))
```

## Submodules

- `cnfsimplifier/`: Advanced CNF manipulation and simplification (optional Rust backend in `cnfsimplifier_rs/`)
- `dissolve/`: Distributed SAT solving with dilemma rules
- `features/`: SAT instance feature extraction and analysis
- `interpolant/`: Boolean interpolation algorithms
- `knowledge_compiler/`: Knowledge compilation to DNNF/OBDD
- `maxsat/`: Maximum satisfiability solvers
- `modal/`: finite-model modal reasoning and bounded countermodel search
- `nnf/`: Negation normal form reasoning
- `backbone/`: Boolean backbone computation
- `prime/`: Prime implicant and prime implicate enumeration
- `qbf/`: Quantified Boolean formula support
- `sat/`: Core SAT solver implementations
