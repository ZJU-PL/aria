Context-Free Language Reachability
===================================

Introduction
=====================

The CFL module (``aria/cfl``) implements algorithms for context-free language (CFL) reachability analysis, a powerful framework for program analysis problems including points-to analysis, interprocedural dataflow analysis, and shape analysis.

Key Features
-------------

* **CFL-Reachability Algorithms**: Classical and quantum-inspired dynamic transitive closure
* **Dyck Language Support**: Balanced-parenthesis reachability for call-return matching
* **Graph-Based Analysis**: Efficient graph algorithms for program analysis
* **Matrix-Based Computation**: Matrix formulations for parallel processing
* **Strong Component Reduction**: Optimization via strongly connected components

Components
=====================

CFL Solvers (``aria/cfl/cfl_solver.py``)
------------------------------------------

Core CFL-reachability solving:

.. code-block:: python

   from aria.cfl import CFLSolver

   # Create solver for CFL-reachability
   solver = CFLSolver()

   # Define grammar and graph
   grammar = load_grammar("grammar.txt")
   graph = load_graph("program_graph.dot")

   # Compute reachability
   reachable = solver.solve(grammar, graph)

Dynamic Transitive Closure (``aria/cfl/cfl_dtc.py``, ``aria/cfl/dtc.py``)
-----------------------------------------------------------------------------

Dynamic transitive closure algorithms including quantum-inspired methods:

.. code-block:: python

   from aria.cfl.cfl_dtc import QuantumDTC

   # Quantum-inspired dynamic transitive closure
   dtc = QuantumDTC()
   closure = dtc.compute(graph)

Grammar Support (``aria/cfl/grammar.py``)
-------------------------------------------

Context-free grammar manipulation:

* Grammar parsing and validation
* Production rule management
* Dyck language generation
* Grammar normalization

Matrix-Based Algorithms (``aria/cfl/matrix.py``, ``aria/cfl/pag_matrix.py``)
-------------------------------------------------------------------------------

Matrix formulations for efficient CFL-reachability:

.. code-block:: python

   from aria.cfl.pag_matrix import PAGMatrix

   # Program assignment graph as matrix
   pag = PAGMatrix(graph)
   result = pag.cfl_reachability(grammar)

Strong Component Reduction (``aria/cfl/sc_solver.py``)
--------------------------------------------------------

Optimization using strongly connected components:

.. code-block:: python

   from aria.cfl.sc_solver import SCReducer

   # Reduce problem via SCC decomposition
   reducer = SCReducer()
   reduced_result = reducer.solve(graph, grammar)

Applications
=====================

* **Points-To Analysis**: Field-sensitive and context-sensitive pointer analysis
* **Call Graph Construction**: Interprocedural control flow analysis
* **Alias Analysis**: May-alias and must-alias analysis
* **Shape Analysis**: Heap shape abstraction
* **Taint Analysis**: Information flow tracking
* **Interprocedural Dataflow**: Summary-based dataflow analysis

References
=====================

- Reps, T. (1998). *Program Analysis via Graph Reachability*. Information and Software Technology
- Yannakakis, M. (1990). *Graph-Theoretic Methods in Database Theory*. PODS 1990
- Zhang, Q., Su, Z. (2017). *Context-Sensitive Data-Dependence Analysis via Linear Conjunctive Language Reachability*. POPL 2017
- Liu, J., et al. (2024). *Dynamic Transitive Closure-Based Static Analysis through the Lens of Quantum Search*. TOSEM 2024
