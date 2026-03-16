C Frontend for EFMC
====================

This document describes the C frontend for EFMC, which allows you to convert C programs (specifically loops) to EFMC transition systems for verification.

Overview
--------

The EFMC C frontend enables you to:

1. Parse C programs and extract loops
2. Convert loops to EFMC transition systems
3. Verify the converted programs using EFMC's verification engines
4. Generate CHC (Constrained Horn Clause) format for external solvers

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from efmc.frontends.c2efmc import c_to_efmc

   # Convert a C file to a transition system
   ts = c_to_efmc("program.c")

   # The transition system can now be used with EFMC engines
   print(f"Variables: {[str(v) for v in ts.variables]}")
   print(f"Initial condition: {ts.init}")
   print(f"Transition relation: {ts.trans}")
   print(f"Post condition: {ts.post}")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.frontends.c2efmc import CToEFMCConverter

   # Create converter instance
   converter = CToEFMCConverter()

   # Parse C file
   ast = converter.parse_c_file("program.c")

   # Convert to transition system (defaults to main function)
   ts = converter.convert_file_to_transition_system("program.c")

   # Or specify a different function
   ts = converter.convert_file_to_transition_system("program.c", function="my_function")

   # Generate CHC format for external verification
   chc_str = ts.to_chc_str()
   with open("program.smt2", "w") as f:
       f.write(chc_str)

Supported C Features
--------------------

The C frontend currently focuses on loop-centric verification:

- **Integer Variables**: Support for integer arithmetic operations
- **Loops**: While loops, for loops
- **Assignments**: Variable assignments
- **Conditions**: Boolean conditions for loops and branches
- **Arithmetic Operations**: Addition, subtraction, multiplication, division, modulo
- **Comparison Operations**: Equality, inequality, less than, greater than, etc.

Limitations
-----------

- **Single Loop**: Currently focuses on programs with a single loop
- **Integer Variables**: Primarily supports integer arithmetic
- **Simple Control Flow**: Works best with structured loops
- **Function Selection**: Defaults to the ``main`` function

Example C Programs
-------------------

Simple Counter
~~~~~~~~~~~~~~

.. code-block:: c

   int main() {
       int x = 10;
       while (x > 0) {
           x = x - 1;
       }
       assert(x == 0);
       return 0;
   }

Fibonacci Computation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   int main() {
       int n = 10;
       int a = 0, b = 1;
       while (n > 0) {
           int temp = a + b;
           a = b;
           b = temp;
           n = n - 1;
       }
       assert(a >= 0);
       assert(b >= 0);
       return 0;
   }

Verification Workflow
--------------------

1. **Parse C Program**: The frontend parses the C file using ``pycparser`` and builds an AST
2. **Extract Function**: Selects the target function (default: ``main``)
3. **Detect Loop**: Identifies loops in the function
4. **Extract Variables**: Collects all variables used in the loop
5. **Build Transition System**: Creates initial conditions, transition relations, and post conditions
6. **Verify**: Uses EFMC engines or external solvers to verify the safety properties

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

   ts = c_to_efmc("program.c")
   chc_str = ts.to_chc_str()

   # Save for external verification
   with open("program.smt2", "w") as f:
       f.write(chc_str)

   # Verify with Z3
   # z3 program.smt2

Command-Line Usage
------------------

You can use the C frontend directly from the command line:

.. code-block:: bash

   efmc --file program.c --lang c --engine ef --template interval

The ``--lang c`` option tells EFMC to use the C frontend. If not specified, EFMC will auto-detect the language based on the file extension.

Future Enhancements
-------------------

- Support for nested loops
- Array and pointer support
- Support for more C language features
- Multiple function verification
- Support for floating-point arithmetic

Error Handling
--------------

The frontend includes robust error handling:

- Graceful parsing failures with detailed error messages
- Warning messages for unsupported features
- Comprehensive logging for debugging

Testing
-------

Test files are provided to validate the implementation:

- ``test_c2efmc.py``: Basic functionality tests

Run tests with:

.. code-block:: bash

   python -m pytest efmc/tests/test_c2efmc.py
