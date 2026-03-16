Symbolic Abstraction
====================

Overview
--------

Symbolic Abstraction is a verification technique that computes the strongest consequence of a formula within a given abstract domain. Unlike template-based approaches that search for invariants by solving constraints, symbolic abstraction iteratively refines an invariant by computing the strongest abstract representation of the current invariant conjoined with the transition relation.

The symbolic abstraction prover in EFMC implements a fixpoint computation that:
1. Starts with the initial condition as the current invariant
2. At each iteration, computes the strongest consequence of the current invariant and transition relation in the chosen abstract domain
3. Takes the disjunction with the previous invariant
4. Continues until a fixpoint is reached
5. Verifies that the computed invariant implies the safety property

Supported Abstract Domains
--------------------------

**Interval Domain:**
- Tracks lower and upper bounds for each variable
- Suitable for integer and real variables
- Also supports bit-vector intervals

**Bits Domain:**
- Tracks known bits for bit-vector variables
- For each bit position, tracks if it must be 0, must be 1, or is unknown
- Provides fine-grained bit-level information

**Known Bits Domain:**
- Similar to bits domain but focuses on known bit patterns
- More efficient than full bits analysis

Algorithm
---------

The symbolic abstraction algorithm computes the least fixpoint:

.. math::

   I_0 = \text{init}
   
   I_{i+1} = I_i \vee \text{strongest\_consequence}(I_i \wedge \text{trans}, \text{domain})

where `strongest_consequence` computes the strongest formula in the abstract domain that is implied by the input formula.

Usage
-----

The symbolic abstraction prover can be used via the command line:

.. code-block:: bash

   efmc --engine symabs --symabs-domain interval --lang chc --file program.smt2

For bit-vector programs with bit-level analysis:

.. code-block:: bash

   efmc --engine symabs --symabs-domain known_bits --lang chc --file program.smt2

Command-Line Options
--------------------

- ``--symabs-domain``: Abstract domain to use
  - ``interval``: Interval abstraction (default)
  - ``bits``: Full bit-level analysis
  - ``known_bits``: Known bits analysis

Programmatic Usage
------------------

.. code-block:: python

   from efmc.engines.symabs import SymbolicAbstractionProver
   from efmc.sts import TransitionSystem
   
   # Create transition system
   sts = TransitionSystem(...)
   
   # Create prover
   prover = SymbolicAbstractionProver(sts)
   prover.domain = 'interval'  # or 'bits', 'known_bits'
   
   # Solve
   result = prover.solve()
   
   if result.is_safe:
       print(f"System is safe")
       print(f"Invariant: {result.invariant}")
   else:
       print("System is unsafe or unknown")

Implementation Details
----------------------

**Bit-Vector Symbolic Abstraction:**
- Uses interval abstraction for bit-vector variables
- Supports signed and unsigned interpretations
- Can track overflow and underflow conditions

**Bit-Level Symbolic Abstraction:**
- Analyzes individual bits of bit-vector variables
- Tracks known bits (must be 0 or 1) vs unknown bits
- More precise but computationally more expensive

**Fixpoint Detection:**
- Uses validity checking to detect when the invariant has converged
- Stops when the new invariant implies the previous one

Advantages
----------

- **Precision**: Can be more precise than template-based approaches for certain domains
- **Automatic Refinement**: Iteratively refines the invariant without manual template selection
- **Domain Flexibility**: Supports multiple abstract domains

Limitations
-----------

- **Computational Cost**: Computing strongest consequences can be expensive
- **Domain Dependency**: Precision depends heavily on the chosen abstract domain
- **Convergence**: May require many iterations to reach fixpoint

Related Work
------------

- Reps, T., Horwitz, S., & Sagiv, S. (1995). Precise interprocedural dataflow analysis via graph reachability. POPL'95.
- Cousot, P., & Cousot, R. (1977). Abstract interpretation: A unified lattice model for static analysis of programs. POPL'77.

