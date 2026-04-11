Translator Module
=================

``aria.translator`` contains format converters between common automated-reasoning
input languages and downstream solver encodings.

Current scope
-------------

The current translator package includes:

**CNF / propositional**

* ``dimacs2smt.py``: DIMACS CNF to SMT-LIB2
* ``cnf2smt.py``: CNF to SMT-LIB2 encoding
* ``cnf2lp.py``: CNF to linear-programming style output
* ``smt2dimacs.py``: SMT-LIB propositional fragment to DIMACS
* ``opb2smt.py``: OPB / WBO-style pseudo-Boolean constraints to SMT-LIB2
* ``wcnf2z3.py``: weighted CNF to Z3 optimization
* ``wcnf2smt.py``: weighted CNF / MaxSAT text to SMT-LIB2 with soft constraints

**QBF**

* ``qbf2smt.py``: QBF to SMT-LIB2
* ``qcir2smt.py``: QCIR to SMT-LIB2

**SMT-LIB and symbolic targets**

* ``smt2c.py``: SMT-LIB to C code generation
* ``smt2sympy.py``: SMT-LIB to SymPy expressions

**SyGuS and FlatZinc**

* ``sygus2smt.py``: SyGuS to SMT-LIB2
* ``fzn2omt/``: FlatZinc-to-OMT translators and model adapters

**Shared infrastructure**

* ``registry.py``: translator capability registry used by the CLI
* ``parsing.py``: shared parsing adapters

Programmatic usage
------------------

.. code-block:: python

   from aria.translator import dimacs2smt

   dimacs2smt.convert_file("input.cnf", "output.smt2")

CLI access
----------

The registry-backed command-line frontend is ``aria.cli.fmldoc_cli``:

.. code-block:: bash

   aria-fmldoc translate -i input.cnf -o output.smt2
   python -m aria.cli.fmldoc_cli formats

Current CLI support covers DIMACS, QDIMACS, QCIR, OPB, WCNF, SyGuS, and selected
SMT-LIB2 conversions.

Notes
-----

The OPB translator supports weighted constraints, ``soft:`` headers,
min/max objectives, richer comparators, and product terms by emitting either
``QF_LIA`` or ``QF_NIA`` SMT-LIB2 as needed.
