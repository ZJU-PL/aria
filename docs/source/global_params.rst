Global Parameters
===============

The ``aria.global_params`` module provides centralized configuration management for the ARIA project. It handles SMT solver discovery, path management, and provides a singleton-based global configuration interface used throughout the codebase.

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
----------------

```
global_params/
├── __init__.py                    (7 lines - module entry point)
└── paths.py                       (206 lines - core implementation)
```

**No subdirectories** - flat module structure.

Module Overview
--------------

### Centralized Configuration System

The global_params module implements a **singleton pattern** configuration manager that:

1. **Manages SMT solver paths** and availability
2. **Provides project-wide constants** for important directories
3. **Implements solver discovery** (local binaries → system PATH)
4. **Offers consistent API** for configuration access across the codebase

**Usage Pattern**: This module is imported in 21 files across the project for solver path resolution and configuration access.

Key Components
------------------

### 1. ``SolverConfig`` Class

A data class representing configuration for a single SMT solver.

**Attributes:**

.. list-table::
   :header: "Attribute, Type, Description"
   :widths: 20, 60

   ``name``, ``str``, Solver identifier (e.g., 'z3', 'cvc5')
   ``exec_name``, ``str``, Executable name (e.g., 'z3', 'cvc5')
   ``exec_path``, ``Path``, Full path to executable
   ``is_available``, ``bool``, Whether solver is installed/accessible

### 2. ``SolverRegistry`` Metaclass

Implements the **singleton pattern** to ensure only one ``GlobalConfig`` instance exists:

* Prevents duplicate initialization
* Provides consistent access across the application
* Thread-safe configuration management

### 3. ``GlobalConfig`` Class (Singleton)

The core configuration manager with comprehensive capabilities.

**Managed Solvers** (5 solvers total):

.. list-table::
   :header: "Solver, Purpose"
   :widths: 15, 60

   ``z3``, Z3 SMT solver
   ``cvc5``, CVC5 solver
   ``mathsat``, MathSAT solver
   ``yices2``, Yices2 SMT solver
   ``sharp_sat``, SharpSAT solver for model counting

**Path Properties:**

* ``PROJECT_ROOT`` - Root directory of ARIA project
* ``BIN_SOLVERS_PATH`` - Path to binary executables (``bin_solvers/``)
* ``BENCHMARKS_PATH`` - Path to benchmark files (``benchmarks/``)

**Key Methods:**

.. list-table::
   :header: "Method, Description"
   :widths: 30, 60

   ``get_solver_path(name)``, Retrieve executable path for a solver
   ``is_solver_available(name)``, Check if solver is installed/accessible
   ``set_solver_path(name, path)``, Set custom solver path
   ``get_smt_solvers_config()``, Get full solver configuration dictionary

**Solver Discovery Logic:**

1. First searches in local ``bin_solvers/`` directory
2. Falls back to system PATH
3. Logs warnings for unavailable solvers
4. Caches results for performance

**Solver Configuration Details:**

From ``get_smt_solvers_config()``:

.. list-table::
   :header: "Solver, Availability, Path Retrieval, Default Args"
   :widths: 15, 20, 25

   ``z3``, ✓, ✓, ``-in``
   ``cvc5``, ✓, ✓, ``-q -i``
   ``mathsat``, ✓, ✓, ``(empty)``

### 4. Module-Level Exports

From ``__init__.py``:

.. code-block:: python

   from aria.global_params import (
       global_config,      # GlobalConfig singleton instance
       SMT_SOLVERS_PATH,  # Dictionary of solver configurations
       PROJECT_ROOT,       # Path object pointing to project root
       BIN_SOLVERS_PATH,  # Path object to bin_solvers directory
       BENCHMARKS_PATH     # Path object to benchmarks directory
   )

Usage Patterns Across Codebase
-------------------------

The module is imported in **21 files** across the project, with common use cases:

### 1. Solver Path Retrieval (Most Common)

.. code-block:: python

   from aria.global_params import global_config
   
   solver_path = global_config.get_solver_path("z3")

**Used in**: efbv_bin_solvers.py, mathsat_solver.py, cvc5_sygus_abduct.py, cvc5_interpolant.py, smtinterpol_interpolant.py, z3_plus_smtlib_solver.py, dimacs_counting.py, optimization/bin_solver.py, eflira_parallel.py

### 2. Availability Checking

.. code-block:: python

   from aria.global_params import global_config
   
   if global_config.is_solver_available("sharp_sat"):
       # Use solver
       pass

**Used in**: test_bv_counting.py, test_bool_counting.py, test_sygus_inv.py, dimacs_counting.py

### 3. SMT Solver Configuration Dictionary

.. code-block:: python

   from aria.global_params import SMT_SOLVERS_PATH
   
   z3_config = SMT_SOLVERS_PATH["z3"]
   solver_bin = z3_config["path"] + " " + z3_config["args"]

**Used in**: pcdclt/solver.py, pcdclt/eval_smt.py, pcdclt/tests/test_solver.py, pcdclt/tests/test_process_cleanup.py

### 4. Benchmark Path Access

.. code-block:: python

   from aria.global_params import BENCHMARKS_PATH
   
   cnf_path = BENCHMARKS_PATH / "dimacs" / "parity_5.cnf"

**Used in**: bool/features/sat_instance.py

Physical Directory Structure
-------------------------

```
bin_solvers/          # Solver binaries
├── z3               # 23.7 MB Z3 executable
├── cvc5             # 25.1 MB CVC5 executable
├── mathsat          # 11.5 MB MathSAT executable
└── sharpSAT         # 1.2 MB SharpSAT executable

benchmarks/          # Test and evaluation benchmarks
├── abduction/
├── chc/
├── dimacs/
├── fzn/
├── qbf/
├── smtlib2/
├── sygus-inv/
└── sygus-pbe/
```

Domains Using Global Parameters
-----------------------------

The module serves as a central hub for:

### SMT Solving
**aria/smt/pcdclt/** - Parallel CDCL(T) solver configuration**

* Solver path resolution
* Configuration dictionary access

### Quantifier Elimination
**aria/quant/efbv/**, **aria/quant/eflira/**, **aria/quant/qe/** - Multiple solver orchestration

* Multiple solver integration
* Custom path support

### Model Counting
**aria/counting/bool/** - SharpSAT integration

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
**aria/bool/features/** - SAT instance feature extraction

* Benchmark path access
* Feature computation infrastructure

Design Patterns
--------------

### 1. Singleton Pattern

Ensures single ``GlobalConfig`` instance via metaclass:

* **Benefits**:
  - Consistent configuration across all imports
  - Prevents initialization conflicts
  - Efficient memory usage

### 2. Lazy Initialization

Solvers located on first access:

* Reduces startup time
* Only searches when needed
* Caches results for performance

### 3. Fallback Strategy

Three-tier discovery:

1. Local ``bin_solvers/`` directory (preferred)
2. System PATH (secondary)
3. Unavailable (logged as warning)

### 4. Logging Integration

Uses Python logging for:

* Unavailable solver warnings
* Configuration errors
* Debug information for troubleshooting

Documentation Quality
------------------

**Strengths:**

* Comprehensive docstrings for all classes and methods
* Clear parameter and return type documentation
* Includes exception documentation
* Explains design rationale (e.g., singleton pattern purpose)

**Coverage:**

* All public methods have docstrings
* Classes have descriptive summaries
* Complex logic (solver discovery) is well documented

Recommendations for Enhancement
------------------------------

Based on the analysis, potential improvements include:

1. **Usage Examples** - Add common usage patterns for each configuration method
2. **Version Documentation** - Document expected solver versions/compatibility
3. **Troubleshooting Guide** - Add section for solving common solver installation issues
4. **Environment Variable Support** - Document environment variables for custom solver paths
5. **Download Script Documentation** - Explain bin_solvers/download.py for initial solver setup

Integration Points
------------------

The module serves as the **central hub** for:

1. **Solver Orchestration**: All external solver invocations route through this module
2. **Path Resolution**: Standardizes project-wide path references
3. **Feature Testing**: Test files check solver availability before running
4. **Parallel Processing**: Used in multi-process contexts (singleton ensures consistency)

Key Features
------------

1. **Singleton Management**: Ensures single GlobalConfig instance
2. **Solver Discovery**: Automatic path resolution with fallback strategy
3. **Path Management**: Centralized project constants (PROJECT_ROOT, BIN_SOLVERS_PATH, BENCHMARKS_PATH)
4. **5 Supported Solvers**: Z3, CVC5, MathSAT, Yices2, SharpSAT
5. **21 Integration Points**: Used across SMT, quantifiers, synthesis, Boolean reasoning
6. **Thread-Safe**: Singleton pattern with metaclass for concurrent access
7. **Logging Integration**: Warnings and errors properly logged
8. **Extensible**: Easy to add new solvers to configuration

Size Statistics
---------------

* Total Python files: 2
* Total Python lines: ~213
* paths.py: 206 lines
* __init__.py: 7 lines
* Managed solvers: 5
* Integration points: 21+
