CLI Tools
=========

The ``aria.cli`` module provides command-line interface tools for various automated reasoning tasks, including EFSMT solving, model counting, optimization (OMT/MaxSAT), UNSAT core and MUS computation, AllSMT model enumeration, and an enhanced SMT server with advanced features.

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
----------------

```
cli/
├── __init__.py                      (0 lines - empty)
├── efsmt.py                         Exists-Forall SMT solver
├── fmldoc.py                        Format conversion and validation
├── mc.py                            Model counting
├── pyomt.py                         OMT / MaxSMT optimization
├── maxsat.py                        MaxSAT (WCNF) solver
├── unsat_core.py                    UNSAT core / MUS / MSS
├── allsmt.py                        All satisfying models enumeration
├── smt_server.py                    Enhanced SMT server (IPC)
├── README.md                        CLI usage guide
└── tests/
    ├── test_smt_server.py
    ├── test_cli_maxsat.py
    ├── test_cli_unsat_core.py
    └── test_cli_allsmt.py
```

**No subdirectories** - flat module structure with tests/ subdirectory.

CLI Tools Overview
------------------

### 1. **efsmt.py** - Exists-Forall SMT Solver

**Purpose**: Solve EFSMT (Exists-Forall Satisfiability Modulo Theories) problems using various algorithms.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.efsmt <file> [options]

**Arguments**:

* ``file`` - EFSMT SMT-LIB2 file (``.smt2``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--parser``, ``z3``, Parsing backend (z3 or sexpr)
   ``--theory``, ``auto``, Theory selection (auto, bool, bv, lira)
   ``--engine``, ``auto``, Solver engine (auto, z3, cegar, efbv-par, efbv-seq, eflira-par, eflira-seq)
   ``--bv-solver``, (from config), Backend for efbv-seq
   ``--forall-solver``, (from config), Binary solver for eflira-par
   ``--max-loops``, (default), Max iterations for CEGAR engines
   ``--timeout``, (default), Timeout in seconds for Z3-based checks
   ``--log-level``, ``INFO``, Logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

* Automatic theory detection (Boolean, BitVector, LIRA)
* Multiple solver engines (Z3, CEGAR, parallel EF-BV/LIRA)
* Support for QF_BV, QF_LIRA, and Boolean theories
* Configurable timeouts and logging

**Theory Support:**

* **Boolean**: Pure SAT problems
* **BitVector**: Quantifier-free bit-vector logic
* **LIRA**: Linear Integer/Real Arithmetic
* **Auto**: Auto-detect from problem content

### 2. **fmldoc.py** - Logic Constraint Format Utilities

**Purpose**: Translate, validate, and analyze supported logic constraint formats.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.fmldoc <command> [options]

**Commands**:

#### ``translate`` - Translate between formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m aria.cli.fmldoc translate -i INPUT_FILE -o OUTPUT_FILE [options]

**Options:**

.. list-table::
   :header: "Option, Description"
   :widths: 25, 60

   ``-i, --input-file``, Input file (required)
   ``-o, --output-file``, Output file (required)
   ``--input-format``, Input format
   ``--output-format``, Output format
   ``--auto-detect``, Auto-detect formats from extensions

#### ``validate`` - Validate file format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m aria.cli.fmldoc validate -i INPUT_FILE [options]

**Options:**

.. list-table::
   :header: "Option, Description"
   :widths: 25, 60

   ``-i, --input-file``, Input file (required)
   ``-f, --format``, File format

#### ``analyze`` - Analyze properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m aria.cli.fmldoc analyze -i INPUT_FILE [options]

**Options:**

.. list-table::
   :header: "Option, Description"
   :widths: 25, 60

   ``-i, --input-file``, Input file (required)
   ``-f, --format``, File format

#### ``formats`` - List supported formats and translations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m aria.cli.fmldoc formats

**Output:**

* Validation and analysis formats currently supported
* Implemented translation pairs

#### ``batch`` - Batch process files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m aria.cli.fmldoc batch -i INPUT_DIR -o OUTPUT_DIR [options]

**Options:**

.. list-table::
   :header: "Option, Description"
   :widths: 25, 60

   ``-i, --input-dir``, Input directory (required)
   ``-o, --output-dir``, Output directory (required)
   ``--input-format``, Input format
   ``--output-format``, Output format

**Currently Supported Formats:**

* DIMACS (``.cnf``): validate, analyze, translate to SMT-LIB2
* SMT-LIB2 (``.smt2``): validate, analyze

**Currently Implemented Translation Pairs:**

* DIMACS -> SMT-LIB2

**Global Options:**

* ``-v, --verbose``: Verbose output
* ``-d, --debug``: Debug output
* ``-h, --help``: Show help message

### 3. **mc.py** - Model Counting Tool

**Purpose**: Count models of formulas in various theories (Boolean, QF_BV, Arithmetic).

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.mc <file> [options]

**Arguments**:

* ``file`` - Formula file (``.smt2``, ``.cnf``, ``.dimacs``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--theory``, ``auto``, Theory type (bool, bv, arith, auto)
   ``--method``, (auto), Counting method (theory-specific)
   ``--timeout``, (default), Timeout in seconds
   ``--log-level``, ``INFO``, Logging level (DEBUG, INFO, WARNING, ERROR)

**Features:**

* Automatic format detection (``.smt2``, ``.cnf``, ``.dimacs``)
* Auto theory detection from content
* **Theory-specific counting methods**:
  - Boolean: DIMACS parallel counting
  - BitVector: Enumeration and sampling
  - Arithmetic: LattE-based and enumeration-based

**Output**: Prints number of models to stdout

### 4. **pyomt.py** - Optimization Problems Solver

**Purpose**: Solve OMT (Optimization Modulo Theory) and MaxSMT problems.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.pyomt <file> [options]

**Arguments**:

* ``file`` - Optimization problem file (``.smt2``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--type``, ``omt``, Problem type (omt, maxsmt)
   ``--theory``, ``auto``, Theory type for OMT (bv, arith, auto)
   ``--engine``, ``qsmt``, Optimization engine (qsmt, maxsat, iter, z3py)
   ``--solver``, (auto), Solver name (engine-specific)
   ``--log-level``, ``INFO``, Logging level (DEBUG, INFO, WARNING, ERROR)

**Problem Types:**

* ``omt``: Optimization Modulo Theory
* ``maxsmt``: Maximum Satisfiability (not yet implemented)

**Theory Support:**

* BitVector (``bv``)
* Arithmetic (``arith``)
* Auto (``auto`` - auto-detect from objective)

**Engines:**

* ``qsmt``: Quantified SMT (default: z3)
* ``maxsat``: MaxSAT solver (default: FM)
* ``iter``: Iterative search (default: z3-ls)
* ``z3py``: Z3 Python API (default: z3py)

**Default Solver Mapping**:

* qsmt → z3
* maxsat → FM
* iter → z3-ls
* z3py → z3py

**Note**: MaxSMT support is not yet implemented in the CLI.

### 5. **maxsat.py** - MaxSAT Solver

**Purpose**: Solve (weighted partial) MaxSAT problems from WCNF files using RC2, FM, or LSU engines.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.maxsat <file> [options]
   aria-maxsat <file> [options]

**Arguments**:

* ``file`` - WCNF formula file (``.wcnf``, ``.cnf``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--solver``, ``rc2``, MaxSAT engine (rc2, fm, lsu)
   ``--timeout``, (none), Timeout in seconds
   ``--print-model``, (off), Print optimal assignment
   ``--log-level``, ``WARNING``, Logging level

**Output**: Prints ``cost: <n>``, optional ``status``, and with ``--print-model`` the satisfying literals.

### 6. **unsat_core.py** - UNSAT Core / MUS / MSS

**Purpose**: Compute one minimal unsatisfiable core or enumerate all MUSes from an SMT-LIB2 formula.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.unsat_core <file> [options]
   aria-unsat-core <file> [options]

**Arguments**:

* ``file`` - SMT-LIB2 formula file (``.smt2``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--algorithm``, ``marco``, Algorithm (marco, musx, optux)
   ``--all-mus``, (off), Enumerate all minimal unsatisfiable subsets (MARCO only)
   ``--timeout``, (none), Timeout in seconds
   ``--no-formulas``, (off), Print only core indices, not assertion formulas

If the formula is satisfiable, the tool reports that and exits successfully. For unsatisfiable formulas, prints one or more cores (assertion indices and optionally the formulas).

### 7. **allsmt.py** - All Satisfying Models

**Purpose**: Enumerate all satisfying models of an SMT-LIB2 formula up to a configurable limit.

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.allsmt <file> [options]
   aria-allsmt <file> [options]

**Arguments**:

* ``file`` - SMT-LIB2 formula file (``.smt2``)

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--solver``, ``z3``, AllSMT backend (z3, pysmt, mathsat)
   ``--limit``, ``100``, Maximum number of models to enumerate
   ``--project``, (all vars), Comma-separated variable names to include
   ``--count-only``, (off), Print only the model count
   ``--verbose``, (off), Print each model in detail

**Output**: Without ``--count-only``, prints each model; with ``--count-only``, a single line with the count.

### 8. **smt_server.py** - Enhanced SMT Server

**Purpose**: IPC-based SMT-LIB2 server with advanced aria features (AllSMT, UNSAT cores, backbone, model counting).

**Usage**:

.. code-block:: bash

   python3 -m aria.cli.smt_server [options]

**Options**:

.. list-table::
   :header: "Option, Default, Description"
   :widths: 25, 60

   ``--input-pipe``, ``/tmp/smt_input``, Path to input pipe
   ``--output-pipe``, ``/tmp/smt_output``, Path to output pipe
   ``--log-level``, ``INFO``, Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Supported Commands**:

#### Basic SMT-LIB2 Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header: "Command, Description"
   :widths: 40, 60

   ``declare-const <name> <sort>``, Declare constant (Int, Bool, Real)
   ``assert <expr>``, Assert expression
   ``check-sat``, Check satisfiability
   ``get-model``, Get satisfying model
   ``get-value <var1> <var2> ...``, Get values of variables
   ``push``, Push new scope
   ``pop``, Pop scope
   ``exit``, Exit server

#### Advanced Commands
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header: "Command, Description"
   :widths: 40, 60

   ``allsmt [:limit=<n>] <var1> <var2> ...``, Enumerate all satisfying models
   ``unsat-core [:algorithm=<alg>] [:timeout=<n>] [:enumerate-all]``, Compute UNSAT cores
   ``backbone [:algorithm=<alg>]``, Compute backbone literals
   ``count-models [:timeout=<n>] [:approximate]``, Count satisfying models
   ``set-option <option> <value>``, Configure server options
   ``help``, Show available commands

#### Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header: "Option, Description"
   :widths: 40, 60

   ``:allsmt-model-limit <n>``, Max models to enumerate (default: 100)
   ``:unsat-core-algorithm <marco|musx|optux>``, UNSAT core algorithm (default: marco)
   ``:unsat-core-timeout <n|none>``, Timeout for UNSAT core (default: none)
   ``:model-count-timeout <n>``, Timeout for model counting (default: 60s)

**Usage Example**:

.. code-block:: bash

   # Start server
   python3 -m aria.cli.smt_server
   
   # In another terminal, send commands
   echo "declare-const x Int" > /tmp/smt_input
   echo "assert (> x 0)" > /tmp/smt_input
   echo "check-sat" > /tmp/smt_input
   cat /tmp/smt_output  # Read response

**Advanced Features:**

* **AllSMT**: Model enumeration with limit support
* **UNSAT Cores**: Computation using MARCO, MUSX, OPTUX algorithms
* **Backbone**: Literal extraction from all models
* **Model Counting**: Exact and approximate counting with timeout
* **Scope Management**: Push/pop for incremental solving
* **Named Pipe IPC**: Unix-style pipe communication

### 9. **README.md** - CLI Documentation

Comprehensive documentation in ``aria/cli/README.md`` covering:

* All CLI tools (fmldoc, mc, pyomt, efsmt, maxsat, unsat_core, allsmt, smt_server)
* Server architecture and IPC mechanisms
* Complete command reference and options
* Usage examples and testing

Testing
-------

**Run all CLI tests**:

.. code-block:: bash

   pytest aria/tests/test_cli_*.py -v

**Per-tool tests**:

* ``test_cli_fmldoc.py`` - Format converter
* ``test_cli_mc.py`` - Model counting
* ``test_cli_efsmt.py`` - EFSMT solver
* ``test_cli_pyomt.py`` - OMT solver
* ``test_cli_maxsat.py`` - MaxSAT solver
* ``test_cli_unsat_core.py`` - UNSAT core / MUS
* ``test_cli_allsmt.py`` - AllSMT enumeration
* ``test_smt_server.py`` - SMT server (declare-const, assert, check-sat, get-model, AllSMT, UNSAT core, model counting, help)

Code Statistics
---------------

.. list-table::
   :header: "File, Description"
   :widths: 30, 50

   efsmt.py, Exists-Forall SMT solver
   fmldoc.py, Format conversion and validation
   mc.py, Model counting
   pyomt.py, OMT / MaxSMT optimization
   maxsat.py, MaxSAT (WCNF) solver
   unsat_core.py, UNSAT core / MUS / MSS
   allsmt.py, All satisfying models enumeration
   smt_server.py, Enhanced SMT server (IPC)

**Documentation**: ``aria/cli/README.md``
**Tests**: ``test_cli_*.py``, ``test_smt_server.py``

Dependencies
-----------

Some tools require additional dependencies that may not be installed:

* **pysat**: Required for EFSMT, MC, and pyomt tools
* **pysmt**: Required for pyomt tool

**Note**: The help commands work even if dependencies are missing, showing the tool's interface and options.

Main API Entry Points
----------------------

All CLI tools can be invoked via Python module syntax:

.. code-block:: bash

   python3 -m aria.cli.<tool_name> [options]

Where ``<tool_name>`` is one of:

* ``efsmt`` - Exists-Forall SMT solver
* ``fmldoc`` - Format translator and validator
* ``mc`` - Model counting
* ``pyomt`` - OMT / MaxSMT optimization solver
* ``maxsat`` - MaxSAT (WCNF) solver (RC2, FM, LSU)
* ``unsat_core`` - UNSAT core / MUS / MSS computation
* ``allsmt`` - Enumerate all satisfying models
* ``smt_server`` - SMT server (IPC)

Key Features
------------

1. **Modular Design**: Each tool is a standalone Python module
2. **Comprehensive CLI**: Consistent argument parsing across all tools
3. **Advanced Features**: SMT server provides AllSMT, UNSAT cores, backbone, counting
4. **Flexible Configuration**: Multiple engines, theories, and algorithms per tool
5. **Testing**: SMT server has comprehensive test suite
6. **Documentation**: README.md provides detailed SMT server usage guide
7. **Logging**: Configurable logging levels for all tools
8. **IPC Support**: Named pipe communication for SMT server integration
