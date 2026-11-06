Boolean Reasoning
=================

The ``aria.bool`` module is a comprehensive toolkit for Boolean reasoning, providing algorithms and tools for SAT solving, MaxSAT optimization, quantified Boolean formulas (QBF), CNF simplification, knowledge compilation, and related logical reasoning tasks.

This page describes the package structure and major APIs. Related solver
workflows such as optimization, backbone computation, UNSAT cores, and
knowledge compilation are grouped alongside it in :doc:`index`.

The module contains roughly 23,000 lines of Python code across its subpackages.

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
-------------------

```
aria/bool/
├── __init__.py                    # Main API exports
├── cnf_simplify.py               # CNF simplification CLI tool
├── pysat_cnf.py                  # CNF manipulation utilities
├── tseitin_converter.py           # DNF to CNF transformation
├── cnfsimplifier/                # CNF simplification framework
├── cnfsimplifier_rs/             # Rust-backed variant of the simplifier
├── sat/                          # SAT solver implementations
├── maxsat/                       # MaxSAT solvers and algorithms
├── nnf/                          # NNF reasoning library (largest submodule)
├── qbf/                          # QBF solver and parsers
├── features/                     # SAT instance feature extraction
├── dissolve/                     # Distributed SAT solver
├── knowledge_compiler/           # DNNF and OBDD compilation
├── interpolant/                  # Boolean interpolation algorithms
├── analysis/                     # CNF/QBF structural analysis
├── backbone/                     # SAT backbone computation
├── encodings/                    # Boolean encodings (cardinality, etc.)
├── modal/                        # Modal logic reasoning
├── prime/                        # Prime implicant/implicate enumeration
└── threeval/                     # Three-valued (Kleene) logic
```

SAT Solvers (``sat/``)
~~~~~~~~~~~~~~~~~~~~~~

Multiple SAT solver backends with unified interface.

**Key Classes:**

* ``PySATSolver``: Wrapper around PySAT supporting 14+ solvers (CDCL, Glucose, MapleSAT, etc.)
  - Methods: ``check_sat()``, ``check_sat_assuming()``, ``get_model()``, ``sample_models()``, ``reduce_models()``
* ``Z3SATSolver``: Z3-based SAT solver implementation
* ``brute_force.py``: pure Python brute-force solvers
  (``solve_sat_brute_force()`` and ``solve_sat_brute_force_parallel()``)

**Features:**

* Support for 14 solver backends (cadical, glucose, lingeling, maple, minisat, etc.)
* Model enumeration and sampling
* Model reduction using dual SAT solver technique
* Parallel solving capabilities
* UNSAT core extraction

MaxSAT Solvers (``maxsat/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve weighted and unweighted MaxSAT optimization problems.

**Key Classes:**

* ``MaxSATSolver``: Main wrapper supporting multiple engines
  - Methods: ``solve()``, ``solve_wcnf()``, ``tacas16_binary_search()``
* ``MaxSATSolverResult`` (``aria.bool.maxsat.maxsat_solver``): Dataclass for
  results (cost, solution, runtime, status); it is not re-exported from
  ``aria.bool.maxsat``
* ``FM``: Fu-Malik algorithm implementation (WMSU1 variant)
* ``RC2``: Relaxable Cardinality Constraints solver (top-ranked in MaxSAT Evaluation)
* ``AnytimeMaxSAT``: Anytime MaxSAT solving

**Features:**

* Weighted and partial MaxSAT support
* Multiple algorithms: FM, RC2, OBV-BS, OBV-BS-Anytime
* TACAS'16 binary search for bit-vector optimization
* Integration with PySAT solvers

NNF Reasoning (``nnf/``)
~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive library for Negation Normal Form manipulation and reasoning.

**Key Classes:**

* ``NNF``: Base abstract class for all NNF sentences (~1,770 lines)
  - Properties: ``decomposable``, ``deterministic``, ``smooth``, ``is_CNF``, ``is_DNF``
  - Methods: ``satisfiable()``, ``valid()``, ``models()``, ``solve()``, ``model_count()``, ``negate()``, ``to_CNF()``, ``condition()``, ``forget()``, ``implicates()``, ``implicants()``
* ``Var``: Variable/literal representation
* ``Aux``: Auxiliary variables (UUID-based)
* ``And``, ``Or``: Internal node types
* ``Internal``: Generic internal node class

**Key Modules:**

* ``tseitin.py``: Tseitin transformation (NNF → CNF)
* ``dimacs.py``: DIMACS format load/dump for SAT and CNF
* ``pysat.py``: PySAT backend integration
* ``kissat.py``: Kissat solver integration
* ``amc.py``: Algebraic Model Counting (WMC, probability, gradient computation)
* ``dsharp.py``: DSHARP d-DNNF compiler interface
* ``operators.py``: Logical operators (xor, nand, nor, iff, implies, etc.)
* ``builders.py``: NNF construction utilities
* ``cli.py``: Command-line interface

**Features:**

* Full NNF manipulation with DAG structure
* Model enumeration (native, PySAT, Kissat backends)
* Efficient operations on d-DNNF (decomposable deterministic NNF)
* Prime implicant/implicate extraction
* Variable forgetting/projection
* Smoothness enforcement
* Visualization (DOT format, Jupyter SVG)
* Knowledge compilation map properties verification

CNF Simplification (``cnfsimplifier/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced CNF formula simplification — see :doc:`cnfsimplifier` for the full
standalone reference.

**Simplification Techniques** (9 methods):

1. Tautology elimination
2. Hidden tautology elimination
3. Asymmetric tautology elimination
4. Subsumption elimination
5. Hidden subsumption elimination
6. Asymmetric subsumption elimination
7. Blocked clause elimination
8. Hidden blocked clause elimination
9. Asymmetric blocked clause elimination

**Key Functions:**

* ``simplify_numeric_clauses()``: Convenience API that applies subsumption
  elimination (see :doc:`cnfsimplifier` for the full pass set)
* ``cnf_subsumption_elimination()``, ``cnf_tautoly_elimination()``, etc.

QBF Support (``qbf/``)
~~~~~~~~~~~~~~~~~~~~~~

Quantified Boolean Formula handling.

**Key Classes:**

* ``QBF``: QBF formula representation with quantifier list
  - Methods: ``solve()``, ``solve_with_skolem()``, ``to_z3()``, ``negate()``
* ``QDIMACSParser``: Parser for QDIMACS format files

**Features:**

* QDIMACS and QCIR format parsing
* Z3 backend integration
* Skolem function extraction
* Variable uniqueness validation

Feature Extraction (``features/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SATzilla-style feature extraction for SAT instance analysis — see
:doc:`features` for the full standalone reference.

**Key Classes:**

* ``SATInstance``: Main class for feature extraction
  - Methods: Extracts 40+ features from CNF files

**Feature Categories:**

* Size features (clauses, variables, ratios)
* Variable-clause graph features (mean/coefficient-of-variation/min/max, entropy)
* Variable graph features
* Balance features (positive/negative ratios)
* Horn clause proximity features (Horn-clause fraction and counts)
* DPLL probing features (unit propagation at depths 1, 4, 16, 64, 256)

The local-search probing, power-law/modularity/fractal structure features, and
eight weighted graphs described in the package README are not implemented by
the current Python code.

**Key Modules:**

* ``dpll.py``: DPLL probing implementation
* ``base_features.py``: Core feature extraction algorithms
* ``active_features.py``: Active feature computation
* ``sat_instance.py``: High-level API

Distributed SAT Solver (``dissolve/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Practical implementation of the Dissolve distributed SAT solver — see
:doc:`dissolve` for the full standalone reference.

**Key Classes:**

* ``Dissolve``: Main distributed solver class
  - Methods: ``solve()``
* ``DissolveConfig``: Configuration for solver parameters
* ``DissolveResult``: Result container

**Features:**

* Asynchronous dilemma-rule splits (Algorithm 3)
* Clause sharing via UBTree-like bucket store
* Variable picking with vote aggregation
* Budget strategies (constant or Luby sequence)
* Distribution strategies (dilemma or portfolio)
* PySAT backend support

Knowledge Compilation (``knowledge_compiler/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile logical formulas into efficient representations.

**Key Classes:**

* ``DNF_Node``: Node in DNNF tree
* ``BDD``: Binary Decision Diagram node
* ``Dtree_Compiler``: Decision tree compilation

**Features:**

* DNNF (Decomposable Negation Normal Form) compilation
* OBDD (Ordered Binary Decision Diagram) compilation
* Decision tree compilation
* Visualization support (DOT format)
* Model counting on compiled structures

Boolean Interpolation (``interpolant/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute Craig interpolants for proof analysis.

**Key Algorithms:**

* Proof-based interpolation (McMillan's method from CAV 2003)
* Core-based interpolation
* PySMT integration

**Key Files:**

* ``proof_based_itp.py``: McMillan's interpolation from resolution proofs
* ``core_based_itp.py``: Core-based interpolation algorithms
* ``pysmt_itp.py``: PySMT solver integration

Usage Examples
--------------

SAT Solving
~~~~~~~~~~~

.. code-block:: python

   from pysat.formula import CNF
   from aria.bool import PySATSolver

   solver = PySATSolver()
   cnf_formula = CNF(from_clauses=[[1, 2], [-1, 2]])
   solver.add_cnf(cnf_formula)
   result = solver.check_sat()
   model = solver.get_model()

MaxSAT Solving
~~~~~~~~~~~~~~

.. code-block:: python

   from pysat.formula import WCNF
   from aria.bool.maxsat import MaxSATSolver

   wcnf_formula = WCNF()
   wcnf_formula.append([1, 2], weight=1)
   wcnf_formula.append([-1, 2], weight=2)
   maxsat = MaxSATSolver(wcnf_formula)
   result = maxsat.solve()  # Returns MaxSATSolverResult

NNF Reasoning
~~~~~~~~~~~~~

.. code-block:: python

   from aria.bool.nnf import Var, And, Or

   a, b = Var('a'), Var('b')
   sentence = (a & b) | (a & ~b)
   is_sat = sentence.satisfiable()

   for model in sentence.models():
       print(model)

CNF Simplification
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.bool.cnfsimplifier import simplify_numeric_clauses

   simplified = simplify_numeric_clauses([[1, 2], [1, -2, 3]])

QBF Solving
~~~~~~~~~~~

.. code-block:: python

   from aria.bool.qbf import QBF, QDIMACSParser

   parser = QDIMACSParser()
   qdimacs_str = """p cnf 2 2
   a 1 0
   e 2 0
   1 2 0
   -1 -2 0
   """
   qbf = parser.parse_qdimacs(qdimacs_str)
   result = qbf.solve()

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pysat.formula import CNF
   from aria.bool.features.sat_instance import SATInstance

   instance = SATInstance(CNF(from_clauses=[[1, 2], [-1, 3]]))
   instance.gen_basic_features()
   # Extract features via instance.features_dict

Dissolve (Distributed SAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pysat.formula import CNF
   from aria.bool.dissolve import Dissolve, DissolveConfig

   cfg = DissolveConfig(k_split_vars=5)
   cnf = CNF(from_clauses=[[1, 2], [-1, 2]])
   res = Dissolve(cfg).solve(cnf)

Main API Entry Points
---------------------

.. code-block:: python

   from aria.bool import PySATSolver, MaxSATSolver
   from aria.bool.cnfsimplifier import simplify_numeric_clauses

   # SAT solving
   from pysat.formula import CNF
   from aria.bool.sat.pysat_solver import PySATSolver
   solver = PySATSolver()
   solver.add_cnf(CNF(from_clauses=[[1, 2], [-1, 2]]))
   result = solver.check_sat()
   model = solver.get_model()

   # MaxSAT solving
   from pysat.formula import WCNF
   from aria.bool.maxsat import MaxSATSolver
   wcnf = WCNF()
   wcnf.append([1, 2], weight=1)
   maxsat = MaxSATSolver(wcnf)
   result = maxsat.solve()  # Returns MaxSATSolverResult

   # NNF reasoning
   from aria.bool.nnf import Var, And, Or
   a, b = Var('a'), Var('b')
   sentence = (a & b) | (a & ~b)
   is_sat = sentence.satisfiable()
   for model in sentence.models():
       print(model)

   # CNF simplification
   from aria.bool.cnfsimplifier import simplify_numeric_clauses
   simplified = simplify_numeric_clauses([[1, 2], [1, -2, 3]])

   # QBF solving
   from aria.bool.qbf import QBF, QDIMACSParser
   parser = QDIMACSParser()
   qbf = parser.parse_qdimacs("""p cnf 2 2\na 1 0\ne 2 0\n1 2 0\n-1 -2 0\n""")
   result = qbf.solve()

   # Feature extraction
   from aria.bool.features.sat_instance import SATInstance
   instance = SATInstance(CNF(from_clauses=[[1, 2], [-1, 3]]))
   instance.gen_basic_features()
   # Extract features via instance.features_dict

   # Dissolve (Distributed SAT)
   from aria.bool.dissolve import Dissolve, DissolveConfig
   cfg = DissolveConfig(k_split_vars=5)
   res = Dissolve(cfg).solve(CNF(from_clauses=[[1, 2], [-1, 2]]))

Key Dependencies
----------------

* **PySAT**: Core SAT solving library
* **Z3**: SMT solver for QBF and additional SAT solving
* **NetworkX**: Graph operations (feature extraction)
* **SciPy** (optional): Entropy/statistics helpers in feature extraction
* **Multiprocessing**: Parallel computing support

Notable Features
----------------

1. **Modular Design**: Clean separation of concerns with dedicated submodules
2. **Multiple Backends**: Support for various SAT/QBF solvers (PySAT, Z3, Kissat)
3. **Knowledge Compilation**: Efficient reasoning through DNNF, OBDD compilation
4. **Scalability**: Distributed solving (Dissolve) for large instances
5. **Comprehensive Testing**: nnf/tests/test_nnf.py with extensive test coverage
6. **CLI Tools**: Command-line interfaces for common operations
7. **Feature Extraction**: SATzilla-style features for machine learning applications

Size Statistics
---------------

* Total Python files: ~100
* Lines of code: ~23,000
* Main NNF library (``nnf/__init__.py``): ~1,850 lines
* Supported SAT solvers: 14+
* Supported MaxSAT algorithms: 4 (FM, RC2, OBV-BS, Anytime)
* Extracted features: 40+
