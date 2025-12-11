Translator Module
===========================

Introduction
=====================

The translator module (``aria/translator``) provides format conversion utilities for various logic constraint representations. It enables translation between different formats used in automated reasoning, including DIMACS, QDIMACS, SMT-LIB2, SyGuS, and others.

Key Features
-------------

* **DIMACS to SMT-LIB2**: Convert CNF formulas to SMT2 format
* **QDIMACS to SMT-LIB2**: Convert quantified Boolean formulas to SMT2 format
* **SMT-LIB2 to C**: Generate C code from SMT formulas
* **SMT-LIB2 to SymPy**: Convert SMT formulas to SymPy expressions
* **SyGuS to SMT-LIB2**: Convert SyGuS synthesis instances to SMT2
* **WCNF to Z3**: Convert weighted CNF to Z3 optimizer instances
* **CNF to Linear Programming**: Convert CNF to LP format
* **FlatZinc to OMT**: Convert FlatZinc to Optimization Modulo Theory formats

Components
=====================

DIMACS to SMT-LIB2 (``aria/translator/dimacs2smt.py``)
-------------------------------------------------------

Converts DIMACS CNF files to SMT-LIB2 format. Supports both command-line usage and programmatic API.

**Command-line usage:**

.. code-block:: bash

   python -m aria.translator.dimacs2smt input.cnf -o output.smt2 -l QF_UF

**Programmatic usage:**

.. code-block:: python

   from aria.translator import dimacs2smt

   output_path = dimacs2smt.convert_dimacs_to_smt2(
       input_path="input.cnf",
       output_path="output.smt2",
       logic="QF_UF",
       var_prefix="v_"
   )

**Features:**
* Supports QF_UF and QF_BV logics
* Customizable variable name prefixes
* Preserves comments from DIMACS files
* Handles empty clauses gracefully

QDIMACS to SMT-LIB2 (``aria/translator/qbf2smt.py``)
------------------------------------------------------

Converts QDIMACS (quantified Boolean formula) files to SMT-LIB2 format using UFBV logic. Uses bit-vector variables to compactly encode multiple Boolean variables.

**Command-line usage:**

.. code-block:: bash

   python -m aria.translator.qbf2smt input.qdimacs

**Features:**
* Converts quantifier prefixes (forall/exists) to SMT quantifiers
* Uses bit-vectors for efficient encoding
* Preserves quantifier structure
* Outputs UFBV logic compatible with SMT solvers

**Example:**

The translator converts QDIMACS quantifier blocks like::

   a 1 2 0
   e 3 4 0

To SMT-LIB2 quantifiers::

   (forall ((vec1 (_ BitVec 2)))
   (exists ((vec2 (_ BitVec 2)))
   ...

CNF to SMT-LIB2 (``aria/translator/cnf2smt.py``)
-------------------------------------------------

Converts CNF formulas to Z3 expressions and SMT-LIB2 format. Provides parsing utilities for DIMACS strings and files.

**Usage:**

.. code-block:: python

   from aria.translator import cnf2smt
   import z3

   # Parse DIMACS file to Z3 expression
   formula = cnf2smt.dimacs_to_z3("input.cnf")

   # Check satisfiability
   solver = z3.Solver()
   solver.add(formula)
   result = solver.check()

**Functions:**
* ``parse_dimacs(filename)``: Parse DIMACS file, returns (num_vars, num_clauses, clauses)
* ``parse_dimacs_string(dimacs_str)``: Parse DIMACS string
* ``dimacs_to_z3(filename)``: Convert DIMACS file to Z3 expression
* ``dimacs_string_to_z3(dimacs_str)``: Convert DIMACS string to Z3 expression
* ``int_clauses_to_z3(clauses)``: Convert integer clause list to Z3 expression

CNF to Linear Programming (``aria/translator/cnf2lp.py``)
-----------------------------------------------------------

Converts CNF formulas to linear programming (LP) format, useful for logic programming applications.

**Usage:**

.. code-block:: python

   from aria.translator import cnf2lp

   cnf2lp.cnf2lp("input.cnf", "output.lp")

**Output format:**

Each clause is converted to an LP rule::

   p1 | p2 :- p3, p4.

Where positive literals appear in the head and negative literals in the body.

SMT-LIB2 to C (``aria/translator/smt2c.py``)
----------------------------------------------

Generates C code from SMT-LIB2 formulas, useful for model counting and solution enumeration.

**Usage:**

.. code-block:: bash

   python -m aria.translator.smt2c input.smt2 -o output.c

**Features:**
* Generates C code for model counting
* Supports bit-vector operations
* Includes modulo arithmetic helper functions
* Can generate SAT checking code

SMT-LIB2 to SymPy (``aria/translator/smt2sympy.py``)
------------------------------------------------------

Converts SMT-LIB2 formulas to SymPy symbolic expressions for symbolic manipulation and analysis.

**Usage:**

.. code-block:: python

   from aria.translator import smt2sympy

   sympy_expr = smt2sympy.translate(smt_formula)

**Features:**
* Preserves formula structure
* Enables symbolic computation with SymPy
* Supports arithmetic and Boolean operations

SyGuS to SMT-LIB2 (``aria/translator/sygus2smt.py``)
------------------------------------------------------

Converts SyGuS (Syntax-Guided Synthesis) instances to SMT-LIB2 format for use with SMT solvers.

**Usage:**

.. code-block:: bash

   python -m aria.translator.sygus2smt --file input.sl

**Conversion process:**
* Replaces ``synth-fun`` with ``declare-fun``
* Converts ``declare-var`` to ``declare-const``
* Transforms ``constraint`` to ``assert``
* Replaces ``check-synth`` with ``check-sat``
* Handles BitVec type annotations

Weighted CNF to Z3 (``aria/translator/wcnf2z3.py``)
----------------------------------------------------

Converts weighted CNF (WCNF) files to Z3 Optimize instances for partial weighted MaxSAT solving.

**Usage:**

.. code-block:: python

   from aria.translator import wcnf2z3
   from z3 import *

   opt, soft_constraints = wcnf2z3.construct_z3_optimizer(open("input.wcnf"))
   result = opt.check()

**Features:**
* Distinguishes hard and soft constraints
* Uses Z3's soft constraint API
* Preserves clause weights
* Supports partial MaxSAT instances

**WCNF format:**
* Hard constraints have weight equal to ``top``
* Soft constraints have weight less than ``top``
* Format: ``p wcnf <vars> <clauses> <top>``

FlatZinc to OMT (``aria/translator/fzn2omt/``)
------------------------------------------------

Converts FlatZinc constraint programming instances to Optimization Modulo Theory (OMT) formats for various solvers.

**Supported solvers:**
* **Z3** (``fzn2z3.py``): Converts to Z3 optimization format
* **CVC4** (``fzn2cvc4.py``): Converts to CVC4 OMT format
* **OptiMathSAT** (``fzn2optimathsat.py``): Converts to OptiMathSAT format

**Common utilities** (``common.py``):
* Shared parsing and conversion functions
* Constraint mapping utilities
* Variable declaration helpers

**Model conversion** (``smt2model2fzn.py``):
* Converts SMT2 model output back to FlatZinc format
* Useful for round-trip conversions

Command-Line Interface (``aria/cli/fmldoc.py``)
================================================

A unified command-line interface for format translation with auto-detection capabilities.

**Translate command:**

.. code-block:: bash

   aria-fmldoc translate -i input.cnf -o output.smt2 --input-format dimacs --output-format smtlib2

**Auto-detect formats:**

.. code-block:: bash

   aria-fmldoc translate -i input.cnf -o output.smt2 --auto-detect

**Supported format pairs:**

* ``dimacs`` → ``smtlib2``
* ``dimacs`` → ``lp``
* ``dimacs`` → ``sympy``
* ``qdimacs`` → ``smtlib2``
* ``sygus`` → ``smtlib2``
* ``smtlib2`` → ``c``
* ``smtlib2`` → ``sympy``
* ``wcnf`` → ``smtlib2``

**Other commands:**

* ``validate``: Validate file format correctness
* ``analyze``: Analyze constraint properties (variable counts, clause counts, etc.)
* ``batch``: Batch process multiple files

**Format detection:**

The CLI can auto-detect formats from file extensions:

* ``.cnf`` → ``dimacs``
* ``.qdimacs`` → ``qdimacs``
* ``.smt2`` → ``smtlib2``
* ``.sy`` → ``sygus``
* ``.lp`` → ``lp``
* ``.fzn`` → ``fzn``

Examples
=====================

Converting DIMACS to SMT-LIB2
-----------------------------

.. code-block:: python

   from aria.translator import dimacs2smt

   # Convert DIMACS CNF to SMT2
   dimacs2smt.convert_dimacs_to_smt2(
       "formula.cnf",
       "formula.smt2",
       logic="QF_UF"
   )

Converting QDIMACS to SMT-LIB2
-------------------------------

.. code-block:: python

   from aria.translator import qbf2smt
   import sys

   # Convert QDIMACS file
   qbf2smt.parse("formula.qdimacs")

Using CNF to Z3 conversion
---------------------------

.. code-block:: python

   from aria.translator import cnf2smt
   import z3

   # Load CNF and check satisfiability
   formula = cnf2smt.dimacs_to_z3("instance.cnf")
   solver = z3.Solver()
   solver.add(formula)

   if solver.check() == z3.sat:
       model = solver.model()
       print("Satisfiable:", model)

Weighted MaxSAT with Z3
-----------------------

.. code-block:: python

   from aria.translator import wcnf2z3
   from z3 import *

   # Load weighted CNF
   with open("instance.wcnf") as f:
       opt, soft = wcnf2z3.construct_z3_optimizer(f)

   # Solve MaxSAT
   if opt.check() == sat:
       model = opt.model()
       print("Optimal solution found")

Applications
=====================

* **Format interoperability**: Convert between different constraint formats
* **Solver integration**: Prepare inputs for various SMT and SAT solvers
* **Model counting**: Generate C code for efficient model enumeration
* **Symbolic analysis**: Convert to SymPy for symbolic manipulation
* **Synthesis**: Convert SyGuS instances for SMT-based synthesis
* **Optimization**: Convert weighted constraints for MaxSAT solving
* **Constraint programming**: Bridge between FlatZinc and OMT solvers

References
=====================

* `DIMACS CNF Format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`_
* `QDIMACS Format <http://www.qbflib.org/qdimacs.html>`_
* `SMT-LIB2 Standard <http://smtlib.cs.uiowa.edu/>`_
* `SyGuS Format <https://sygus.org/>`_
* `FlatZinc Format <https://www.minizinc.org/doc-2.5.5/en/flattening.html>`_
