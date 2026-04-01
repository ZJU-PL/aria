Program Synthesis
===========================

Introduction
=====================

The synthesis module (``aria/synthesis``) provides program synthesis capabilities using constraint solving and inductive techniques. It supports syntax-guided synthesis (SyGuS), programming by example (PBE), and version space algebra.

Key Features
-------------

* **SyGuS Solving**: Syntax-Guided Synthesis for invariant and PBE problems
* **Programming by Example**: Generate programs from input-output examples
* **Version Space Algebra**: Algebraic framework for program space manipulation
* **Multi-Theory Support**: LIA, bit-vectors, strings, and data structures
* **Spyro Framework**: Property synthesis for functional programs

Components
=====================

SyGuS Solvers
-------------

Syntax-Guided Synthesis for invariants and PBE:

.. code-block:: python

   from aria.synthesis import SyGuSInvariantSolver, SyGuSPBESolver

   # Invariant synthesis
   inv_solver = SyGuSInvariantSolver()
   invariant = inv_solver.synthesize(
       sygus_file="benchmarks/sygus-inv/array.sl"
   )

   # Programming by example
   pbe_solver = SyGuSPBESolver()
   program = pbe_solver.synthesize(
       sygus_file="benchmarks/sygus-pbe/phone.sl"
   )

Programming by Example (``aria/synthesis/pbe``)
-------------------------------------------------

Synthesize programs from input-output examples:

.. code-block:: python

   from aria.synthesis.pbe import PBESolver

   # Create solver
   solver = PBESolver(max_expression_depth=3, timeout=30.0)

   # Define examples
   examples = [
       {"x": 1, "y": 2, "output": 3},
       {"x": 3, "y": 4, "output": 7},
       {"x": 5, "y": 1, "output": 6},
   ]

   # Synthesize program
   result = solver.synthesize(examples)
   print(f"Synthesized: {result.expression}")

Version Space Algebra (``aria/synthesis/vsa``)
------------------------------------------------

Algebraic manipulation of program spaces:

.. code-block:: python

   from aria.synthesis.pbe.vsa import VersionSpace, VSAlgebra
   from aria.synthesis.pbe.expressions import var, const, add

   # Create expressions
   x = var("x", Theory.LIA)
   expr1 = add(x, const(5))
   expr2 = add(x, const(10))

   # Build version space
   vs = VersionSpace({expr1, expr2})

   # Filter by examples
   algebra = VSAlgebra()
   filtered = algebra.filter_consistent(vs, examples)

Spyro: Property Synthesis (``aria/synthesis/spyro``)
------------------------------------------------------

Synthesize algebraic properties for functional programs:

.. code-block:: python

   from aria.synthesis.spyro import Spyro

   # Define program specification
   spec_file = "examples/spec/list/append.sp"

   # Synthesize properties
   spyro = Spyro()
   properties = spyro.synthesize(spec_file)

SMT-Based PBE (``aria/synthesis/pbe/``)
----------------------------------------------------

Integration with SMT solvers for synthesis:

* Expression encoding to SMT
* SMT-based verification of synthesized programs
* Counterexample-guided refinement

Applications
=====================

* Automated program repair
* Test case generation from specifications
* API usage pattern synthesis
* Loop invariant generation
* Functional program property inference
* String transformation synthesis

References
=====================

- Gulwani, S. (2011). *Automating String Processing in Spreadsheets*. POPL 2011
- Lau, T. et al. (2003). *Programming by Demonstration using Version Space Algebra*
- Mitchell, T. M. (1982). *Generalization as Search*. Artificial Intelligence
