Context-Free Language Reachability
==================================

Introduction
------------

The CFL module (``aria/automata/cfl``) implements algorithms for context-free
language (CFL) reachability analysis, a powerful framework for program analysis
problems including points-to analysis, interprocedural dataflow analysis, and
shape analysis.

Key Features
------------

* **CFL-Reachability Algorithms**: Classical and matrix-based dynamic reachability
* **Dyck Language Support**: Balanced-parenthesis reachability for call-return matching
* **Graph-Based Analysis**: Efficient graph algorithms for program analysis
* **Matrix-Based Computation**: Matrix formulations for parallel processing
* **Strong Component Reduction**: Optimization via strongly connected components

Components
==========

CFL Solvers (``aria/automata/cfl/cfl_solver.py``)
-------------------------------------------------

Core CFL-reachability solving:

.. code-block:: python

   from aria.automata.cfl import CFLSolver

   solver = CFLSolver()

Dynamic Transitive Closure (``aria/automata/cfl/cfl_dtc.py``, ``dtc.py``)
--------------------------------------------------------------------------

Dynamic transitive closure algorithms (optional quantum-inspired Grover search
via ``qiskit``):

.. code-block:: text

   from aria.automata.cfl.cfl_dtc import RandomCFLSolver
   # Note: requires the optional qiskit package, which ARIA does not install
   # by default (pip install qiskit); the module fails to import without it.

Grammar Support (``aria/automata/cfl/grammar.py``)
--------------------------------------------------

Context-free grammar manipulation:

* Grammar parsing and validation
* Production rule management
* Dyck language generation
* Grammar normalization

Matrix-Based Algorithms (``aria/automata/cfl/matrix.py``, ``pag_matrix.py``)
----------------------------------------------------------------------------

Matrix formulations for efficient CFL-reachability:

.. code-block:: python

   from aria.automata.cfl import PAG_Matrix

   pag = PAG_Matrix()
   pag.add_vertex("var1")
   pag.add_vertex("var2")
   pag.add_edge("var1", "var2", "assign")

Strong Component Reduction (``aria/automata/cfl/sc_solver.py``)
---------------------------------------------------------------

Optimization using strongly connected components:

.. code-block:: python

   from aria.automata.cfl import SCSolver

   solver = SCSolver()

Applications
============

* **Points-To Analysis**: Field-sensitive and context-sensitive pointer analysis
* **Call Graph Construction**: Interprocedural control flow analysis
* **Alias Analysis**: May-alias and must-alias analysis
* **Shape Analysis**: Heap shape abstraction
* **Taint Analysis**: Information flow tracking
* **Interprocedural Dataflow**: Summary-based dataflow analysis

References
==========

- Reps, T. (1998). *Program Analysis via Graph Reachability*. Information and Software Technology
- Yannakakis, M. (1990). *Graph-Theoretic Methods in Database Theory*. PODS 1990
- Zhang, Q., Su, Z. (2017). *Context-Sensitive Data-Dependence Analysis via Linear Conjunctive Language Reachability*. POPL 2017
- Liu, J., et al. (2024). *Dynamic Transitive Closure-Based Static Analysis through the Lens of Quantum Search*. TOSEM 2024
