Boogie Frontend for EFMC
==========================

This document describes the Boogie frontend for EFMC, which allows you to convert Boogie programs (specifically loops) to EFMC transition systems for verification.

Overview
--------

Microsoft Boogie is an intermediate verification language that serves as a common target for multiple program verifiers. The EFMC Boogie frontend enables you to:

1. Parse Boogie programs and extract loops
2. Convert loops to EFMC transition systems
3. Verify the converted programs using EFMC's verification engines
4. Generate CHC (Constrained Horn Clause) format for external solvers

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from efmc.frontends.boogie2efmc import boogie_to_efmc

   # Convert a Boogie file to a transition system
   ts = boogie_to_efmc("program.bpl")

   # The transition system can now be used with EFMC engines
   print(f"Variables: {[str(v) for v in ts.variables]}")
   print(f"Initial condition: {ts.init}")
   print(f"Transition relation: {ts.trans}")
   print(f"Post condition: {ts.post}")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.frontends.boogie2efmc import BoogieToEFMCConverter

   # Create converter instance
   converter = BoogieToEFMCConverter()

   # Parse and extract loops
   ast_program = converter.parse_boogie_file("program.bpl")
   loops, bbs = converter.extract_loops_from_program(ast_program)

   # Convert specific loop to transition system
   ts = converter.convert_loop_to_transition_system(loops[0], bbs)

   # Generate CHC format for external verification
   chc_str = ts.to_chc_str()
   with open("program.smt2", "w") as f:
       f.write(chc_str)

Supported Boogie Features
-------------------------

Statements
~~~~~~~~~~

- **Assignments**: ``x := expr;``
- **Assumptions**: ``assume expr;``
- **Assertions**: ``assert expr;``
- **Havoc**: ``havoc x, y;`` (non-deterministic assignment)
- **Goto**: ``goto label1, label2;``
- **Labels**: ``label:``

Expressions
~~~~~~~~~~~

- **Arithmetic**: ``+``, ``-``, ``*``, ``div``, ``mod``
- **Comparison**: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
- **Boolean**: ``&&``, ``||``, ``==>``, ``!``
- **Literals**: integers, ``true``, ``false``
- **Variables**: identifiers

Data Types
~~~~~~~~~~

- Currently supports integer variables (``int``)
- Boolean expressions for conditions

Example Boogie Programs
-----------------------

Simple Counter
~~~~~~~~~~~~~~

.. code-block:: boogie

   implementation Counter()
   {
       var x, count: int;
       
       entry:
           assume x >= 0;
           count := 0;
           goto loop_header;
       
       loop_header:
           goto loop_body, loop_exit;
       
       loop_body:
           assume x > 0;
           x := x - 1;
           count := count + 1;
           goto loop_header;
       
       loop_exit:
           assume x <= 0;
           assert count >= 0;
           return;
   }

Fibonacci Computation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: boogie

   implementation Fibonacci()
   {
       var n, a, b, temp: int;
       
       entry:
           assume n >= 0;
           a := 0;
           b := 1;
           goto loop_header;
       
       loop_header:
           goto loop_body, loop_exit;
       
       loop_body:
           assume n > 0;
           temp := a + b;
           a := b;
           b := temp;
           n := n - 1;
           goto loop_header;
       
       loop_exit:
           assume n <= 0;
           assert a >= 0;
           assert b >= 0;
           return;
   }

Verification Workflow
---------------------

1. **Parse Boogie Program**: The frontend parses the Boogie file and builds an AST
2. **Extract Basic Blocks**: Convert the AST to a control flow graph of basic blocks
3. **Detect Loops**: Identify loops in the control flow graph
4. **Extract Variables**: Collect all variables used in the loop
5. **Build Transition System**: Create initial conditions, transition relations, and post conditions
6. **Verify**: Use EFMC engines or external solvers to verify the safety properties

Generated Transition System
----------------------------

The converter creates a transition system with:

- **Variables**: Current state variables (e.g., ``x``, ``y``)
- **Prime Variables**: Next state variables (e.g., ``x!``, ``y!``)
- **Initial Condition**: Entry condition for the loop
- **Transition Relation**: How variables change in one loop iteration
- **Post Condition**: Safety properties to verify (from assertions)

Integration with EFMC Engines
------------------------------

The generated transition systems can be used with various EFMC verification engines:

- **Template-based (EF)**: Constraint-based invariant generation
- **PDR/IC3**: Property-directed reachability analysis
- **K-induction**: Inductive verification
- **Houdini**: Iterative invariant inference

CHC Format Generation
---------------------

The frontend can generate CHC (Constrained Horn Clause) format compatible with SMT solvers:

.. code-block:: python

   ts = boogie_to_efmc("program.bpl")
   chc_str = ts.to_chc_str()

   # Save for external verification
   with open("program.smt2", "w") as f:
       f.write(chc_str)

   # Verify with Z3
   # z3 program.smt2

Limitations
-----------

- **Single Loop**: Currently focuses on programs with a single loop
- **Integer Variables**: Primarily supports integer arithmetic
- **Simple Control Flow**: Works best with structured loops
- **Manual Loop Detection**: Falls back to heuristic loop detection when automatic detection fails

Future Enhancements
-------------------

- Support for nested loops
- Array and pointer support  
- More sophisticated loop detection
- Integration with more Boogie language features
- Support for concurrent programs

Error Handling
--------------

The frontend includes robust error handling:

- Graceful parsing failures with detailed error messages
- Fallback loop detection when automatic detection fails
- Warning messages for unsupported features
- Comprehensive logging for debugging

Testing
-------

Test files are provided to validate the implementation:

- ``test_boogie_converter.py``: Basic functionality tests
- ``example_boogie_verification.py``: End-to-end verification examples

Run tests with:

.. code-block:: bash

   python test_boogie_converter.py
   python example_boogie_verification.py

