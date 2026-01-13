Utilities
=========

The ``aria.utils`` module provides common helper functions, type definitions, exception handling, and utility classes used throughout the ARIA codebase. It serves as a foundation for all other modules, offering standardized interfaces for solvers, expressions, values, and file parsing.

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
----------------

```
utils/
├── __init__.py                     (7 lines - main entry point)
├── types.py                         (Type definitions)
├── exceptions.py                    (Exception hierarchy)
├── (Other utility modules across utils/)
└── (Various subdirectories for specialized utilities)
```

**Note**: The utils module is distributed across multiple subdirectories. Key utilities are documented below.

Module-Level Exports
-----------------

From ``__init__.py``:

.. code-block:: python

   from aria.utils import (
       SExprParser,          # S-expression parser
       SolverResult,          # Unified solver result type
       RE_GET_EXPR_VALUE_ALL  # Regex pattern for extracting values
   )

Key Components
--------------

### 1. Type Definitions (``types.py``)

Provides common type definitions used throughout ARIA.

**Key Types:**

* **SolverResult** (enum): Unified solver result type
  - ``SAT``: Formula is satisfiable
  - ``UNSAT``: Formula is unsatisfiable
  - ``UNKNOWN``: Solver couldn't determine status
  - ``ERROR``: Solver error occurred

* **OSType** (enum): Operating system type for solver binaries
* **BinarySMTSolverType** (enum): Types of binary SMT solvers

**Usage Pattern**:

.. code-block:: python

   from aria.utils.types import SolverResult
   
   result = SolverResult.SAT
   if result == SolverResult.SAT:
       print("Satisfiable")

### 2. Exception Hierarchy (``exceptions.py``)

Custom exception hierarchy for ARIA-specific errors.

**Base Exception:**

* **AriaException**: Base exception class for all ARIA errors
  - Provides consistent error handling across modules
  - Supports chaining for error context

**Usage Pattern**:

.. code-block:: python

   from aria.utils.exceptions import AriaException
   
   try:
       # Some aria operation
       pass
   except AriaException as e:
       print(f"ARIA error: {e}")

### 3. S-Expression Parser (``SExprParser``)

Parses S-expression format commonly used in SMT-LIB and related formats.

**Key Methods:**

* ``parse(string)``: Parse S-expression string into Python objects
* Supports nested expressions
* Handles quoted strings and special characters

**Usage Pattern**:

.. code-block:: python

   from aria.utils import SExprParser
   
   parser = SExprParser()
   parsed = parser.parse("(and x y z)")
   # Returns nested Python structure

**Use Cases**:

* **SMT-LIB2 parsing**: Read S-expressions from SMT-LIB format files
* **Formula parsing**: Convert S-expression to internal representations
* **Communication protocols**: Parse solver output in S-expression format

### 4. Solver Utilities

Various utilities for working with external solvers:

**Common Patterns:**

* Binary solver execution and output parsing
* Standardized error handling across different solver backends
* File I/O for solver communication
* Process management for parallel solver execution

### 5. Expression Utilities

Helpers for working with Z3 and SMT expressions:

**Common Functions:**

* Variable extraction from formulas
* Expression transformation and simplification
* Type checking and validation
* Substitution and evaluation utilities

**Related Modules:**

* ``z3_expr_utils.py``: Z3-specific expression utilities
* ``z3_solver_utils.py``: Z3 solver wrapper utilities
* ``pysmt_solver.py``: PySMT solver interface

### 6. Value Conversion

Utilities for converting between different value representations:

**Use Cases:**

* Converting solver outputs to ARIA internal types
* Handling different numeric representations (integers, bit-vectors, reals)
* String/bytes parsing and conversion

Usage Across Codebase
----------------------

The utils module is used extensively throughout ARIA in these domains:

### SMT Solving
**aria/smt/pcdclt/** - Parallel CDCL(T) solver**

* Solver configuration access
* Type checking
* Process management

### Quantifier Solving
**aria/quant/efbv/**, **aria/quant/eflira/**, **aria/quant/qe/** - Multiple solver orchestration

* Binary solver invocation
* Custom path configuration
* Multi-solver coordination

### Model Counting
**aria/counting/bool/** - Boolean model counting

* Solver availability checking
* Model counter invocation

### Synthesis
**aria/synthesis/cvc5/** - SyGuS synthesis with CVC5

* Solver path access
* Binary management

### Abduction
**aria/abduction/** - Abductive reasoning

* Solver orchestration
* Multi-solver support

### Interpolation
**aria/interpolant/** - Interpolant generation

* Multiple solver backends
* Path configuration

### Boolean Reasoning
**aria/bool/** - Boolean logic operations

* SAT solver wrappers
* Expression utilities

### General Utilities
**Expression parsing**, **file I/O**, **process management**

Design Patterns
--------------

### 1. Centralized Configuration

Utils provide standardized access to:

* Solver paths and availability
* Type definitions
* Common interfaces

**Benefits:**

* **Consistency**: Single source of truth for solver configuration
* **Maintainability**: Changes propagate automatically
* **Testing**: Easy to mock utilities for testing

### 2. Error Handling

Unified exception hierarchy:

* ``AriaException`` base class
* Specific exceptions can inherit from it
* Consistent error catching across modules

### 3. Type Safety

Strong typing with enums:

* ``SolverResult`` for clear status communication
* ``OSType`` for platform-specific behavior
* ``BinarySMTSolverType`` for solver categorization

### 4. Parser Utilities

**SExprParser** for S-expression parsing:

* Standard format in SMT and logic communities
* Robust parsing with error handling
* Support for nested structures

Usage Examples
--------------

### Solver Result Type
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.utils.types import SolverResult
   
   def solve_formula(formula):
       # ... solver logic ...
       return SolverResult.SAT
   
   def check_result(result):
       if result == SolverResult.SAT:
           print("Formula is satisfiable")
       elif result == SolverResult.UNSAT:
           print("Formula is unsatisfiable")
       else:
           print("Unknown or error")

### Exception Handling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.utils.exceptions import AriaException
   
   def some_aria_function():
       try:
           # Operation that might fail
           result = perform_solver_call()
       except AriaException as e:
           print(f"ARIA operation failed: {e}")
           raise

### S-Expression Parsing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.utils import SExprParser
   
   # Parse S-expression
   parser = SExprParser()
   parsed = parser.parse("(define-fun my-fun (x) (ite (is-sat x) true false))")
   
   # Access parsed structure
   print(parsed)

Key Features
------------

1. **Type Safety**: Strong typing with enums for results and configurations
2. **Error Handling**: Centralized exception hierarchy for consistent error management
3. **Parser Support**: S-expression parser for SMT-LIB compatibility
4. **Solver Integration**: Utilities for working with external solver binaries
5. **Expression Utilities**: Helpers for expression manipulation and analysis
6. **Value Conversion**: Standardized conversion between different representations
7. **Extensibility**: Easy to add new utilities as needed

Integration Points
------------------

The utils module is imported in **numerous files across ARIA**:

* **SMT solving**: aria/smt/pcdclt/
* **Quantifiers**: aria/quant/efbv/, aria/quant/eflira/
* **Model counting**: aria/counting/bool/
* **Synthesis**: aria/synthesis/cvc5/
* **Abduction**: aria/abduction/
* **Interpolation**: aria/interpolant/
* **Boolean reasoning**: aria/bool/
* **Testing**: aria/tests/

Design Philosophy
-----------------

**"Utility First"**: Provide small, focused, well-tested utilities that:

1. **Do one thing well** - Each utility has a clear, single purpose
2. **Be reusable** - Designed for use across multiple modules
3. **Stay simple** - Avoid unnecessary complexity
4. **Document thoroughly** - Clear docstrings explain purpose and usage
5. **Handle errors gracefully** - Proper exception raising and catching
6. **Standardize interfaces** - Consistent APIs for similar functionality

Notes
-----

* The utils module is the foundation upon which all other ARIA modules build
* Changes to utils should be made carefully as they have wide-ranging effects
* Priority is on correctness and reliability over performance
* Documentation is essential due to the module's foundational role

Size Statistics
---------------

* Total documented components: 6+ (from analysis)
* Type definitions: 3+ enums
* Core utilities: SExprParser, exception hierarchy, solver wrappers
* Integration points: 15+ modules across ARIA
