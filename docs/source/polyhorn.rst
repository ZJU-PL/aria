PolyHorn: Polynomial Horn Clause Solver
======================================

=========
Introduction
=========

PolyHorn is a specialized solver for polynomial Horn clauses that leverages positive semidefinite programming and sum-of-squares optimization techniques. It provides automated verification and satisfiability checking for systems involving polynomial constraints with universal quantification.

The module implements several fundamental theorems from real algebraic geometry and optimization theory:

- **Farkas' Lemma**: For linear polynomial constraints
- **Handelman's Theorem**: For polynomial constraints with linear left-hand sides  
- **Putinar's Theorem**: For general polynomial constraints using sum-of-squares representations

Key Features
-----------

1. **Multiple Theorem Support**: Automatic selection or manual specification of verification theorems
2. **Flexible Input Formats**: Support for SMT-LIB2 format and human-readable syntax
3. **Configurable Degrees**: Control over polynomial degree bounds for optimization
4. **Heuristic Optimizations**: UNSAT core iteration and constant elimination heuristics
5. **Multiple Solver Backends**: Integration with Z3, CVC4, and other SMT solvers

Applications
-----------

- Safety property checking for hybrid systems
- Template-based invariant synthesis


=========
Core Concepts
=========

Horn Clauses
-----------

PolyHorn works with polynomial Horn clauses of the form:

.. math::

   \forall x_1, \ldots, x_n : (P_1(x) \land \ldots \land P_k(x)) \Rightarrow Q(x)

where :math:`P_i(x)` and :math:`Q(x)` are polynomial constraints over real variables.

Verification Theorems
-------------------

**Farkas' Lemma**
   Used for linear polynomial constraints. States that a system of linear inequalities has no solution if and only if there exists a non-negative linear combination that yields a contradiction.

**Handelman's Theorem**  
   Extends Farkas' lemma to polynomial constraints where the left-hand side consists of linear constraints. Uses products of constraint polynomials up to a specified degree.

**Putinar's Theorem**
   Most general approach using sum-of-squares representations. Applicable to arbitrary polynomial constraints by representing positive polynomials as sums of squares.

=========
API Reference
=========

Main Interface
-------------

.. py:function:: execute(formula, config)

   Execute PolyHorn on a formula with the given configuration.
   
   :param formula: Either a path to an SMT2 file or a pysmt.Solver object
   :type formula: Union[str, pysmt.solvers.solver.Solver]
   :param config: Configuration dictionary or path to config file
   :type config: Union[str, dict]
   :returns: Tuple of satisfiability result and model
   :rtype: Tuple[str, dict]
   
   **Example:**
   
   .. code-block:: python
   
      from aria.quant.polyhorn.main import execute
      
      config = {
          "theorem_name": "farkas",
          "solver_name": "z3"
      }
      
      result, model = execute("problem.smt2", config)
      print(f"Result: {result}")  # 'sat', 'unsat', or 'unknown'

Configuration Options
-------------------

The configuration dictionary supports the following options:

.. code-block:: python

   config = {
       "theorem_name": "auto",              # "farkas", "handelman", "putinar", or "auto"
       "solver_name": "z3",                 # Backend SMT solver
       "SAT_heuristic": False,              # Enable satisfiability constraints
       "degree_of_sat": 0,                  # Max degree for SAT constraints
       "degree_of_nonstrict_unsat": 0,      # Max degree for non-strict UNSAT
       "degree_of_strict_unsat": 0,         # Max degree for strict UNSAT  
       "max_d_of_strict": 0,                # Degree for strict case variables
       "unsat_core_heuristic": False,       # Enable UNSAT core iteration
       "integer_arithmetic": False,         # Use integer vs real arithmetic
       "output_path": "output.smt2"         # Path for generated SMT file
   }

Core Classes
-----------

**PositiveModel**
   Main class that manages Horn clause constraints and generates verification conditions.

**Parser**  
   Handles parsing of SMT-LIB2 and human-readable input formats.

**Farkas, Handelman, Putinar**
   Implementation classes for the respective verification theorems.

**Solver**
   Utility class with static methods for constraint manipulation and SMT generation.

=========
Usage Examples
=========

Basic Usage with PySMT
---------------------

.. code-block:: python

   from pysmt.shortcuts import (GE, GT, LE, And, Equals, ForAll, Implies, 
                                Minus, Real, Solver, Symbol)
   from pysmt.typing import REAL
   from aria.quant.polyhorn.main import execute

   # Create symbols
   x = Symbol("x", REAL)
   y = Symbol("y", REAL) 
   z = Symbol("z", REAL)
   l = Symbol("l", REAL)

   # Create solver and add constraints
   solver = Solver(name="z3")
   solver.add_assertion(z < y)
   
   # Add universally quantified Horn clause
   solver.add_assertion(ForAll([l], 
       Implies(
           And(Equals(x, l), GE(x, Real(1)), LE(x, Real(3))),
           And(LE(x, Minus(Real(10), z)), 
               Equals(l, Real(2)),
               Equals(z, Minus(x, y)),
               GT(z, Real(-1)))
       )))

   # Configure and solve
   config = {
       "theorem_name": "farkas",
       "solver_name": "z3"
   }
   
   result, model = execute(solver, config)
   print(f"Satisfiability: {result}")
   if result == 'sat':
       for var, value in model.items():
           print(f"{var}: {value}")

SMT-LIB2 File Input
-----------------

.. code-block:: python

   from aria.quant.polyhorn.main import execute

   # Load from SMT2 file
   config = {
       "theorem_name": "handelman", 
       "degree_of_sat": 2,
       "solver_name": "z3"
   }
   
   result, model = execute("constraints.smt2", config)

Human-Readable Format
-------------------

PolyHorn also supports a more readable input format:

.. code-block:: text

   Program_var: x y z;
   Template_var: a b c;
   
   Precondition: x >= 0 AND y >= 0
   
   Horn_clause: (x >= 1 AND y >= 1) -> (x + y >= 2)
   Horn_clause: (x^2 + y^2 <= 1) -> (x + y <= 2)

Advanced Configuration
--------------------

.. code-block:: python

   # Advanced configuration with degree bounds and heuristics
   config = {
       "theorem_name": "putinar",
       "solver_name": "z3", 
       "SAT_heuristic": True,
       "degree_of_sat": 4,
       "degree_of_nonstrict_unsat": 2,
       "degree_of_strict_unsat": 2,
       "max_d_of_strict": 1,
       "unsat_core_heuristic": True,
       "integer_arithmetic": False
   }
   
   result, model = execute(formula, config)

=========
Implementation Details
=========

Theorem Selection
---------------

When ``theorem_name`` is set to ``"auto"``, PolyHorn automatically selects the appropriate theorem:

1. **Farkas**: If both LHS and RHS constraints are linear
2. **Handelman**: If LHS constraints are linear but RHS may be polynomial  
3. **Putinar**: For general polynomial constraints

Degree Bounds
------------

The degree parameters control the complexity vs. completeness trade-off:

- Higher degrees increase completeness but also computational cost
- ``degree_of_sat``: Controls template complexity for satisfiability
- ``degree_of_*_unsat``: Controls template complexity for unsatisfiability proofs

Heuristic Optimizations
---------------------

**UNSAT Core Iteration**
   Iteratively refines the constraint set by analyzing unsatisfiable cores, potentially reducing solver complexity.

**Constant Elimination**  
   Removes equality constraints involving only constants to simplify the constraint system.

=========
Limitations and Considerations
=========

1. **Decidability**: The underlying problem is generally undecidable; PolyHorn provides a semi-decision procedure
2. **Degree Bounds**: Completeness depends on choosing appropriate degree bounds
3. **Scalability**: Performance degrades with increasing polynomial degrees and number of variables
4. **Solver Dependencies**: Requires external SMT solvers (Z3, CVC4, etc.)

=========
Related Work
=========

- **Sum-of-Squares Programming**: Parrilo, P. A. (2003). Semidefinite programming relaxations for semialgebraic problems.
- **Handelman's Theorem**: Handelman, D. (1988). Representing polynomials by positive linear functions on compact convex polyhedra.
- **Putinar's Positivstellensatz**: Putinar, M. (1993). Positive polynomials on compact semi-algebraic sets.
- **Polynomial Invariants**: Sankaranarayanan, S., Sipma, H. B., & Manna, Z. (2004). Non-linear loop invariant generation using Gröbner bases.

=========
References
=========

- Farkas, J. (1902). Theorie der einfachen Ungleichungen. Journal für die reine und angewandte Mathematik.
- Handelman, D. (1988). Representing polynomials by positive linear functions on compact convex polyhedra. Pacific Journal of Mathematics.
- Putinar, M. (1993). Positive polynomials on compact semi-algebraic sets. Indiana University Mathematics Journal.
- Parrilo, P. A. (2003). Semidefinite programming relaxations for semialgebraic problems. Mathematical Programming.
