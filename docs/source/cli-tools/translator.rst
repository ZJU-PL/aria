Translator Module
=================

``aria.utils.translator`` contains format converters between common automated-reasoning
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

   from aria.utils.translator.dimacs2smt import convert_dimacs_to_smt2

   convert_dimacs_to_smt2("input.cnf", "output.smt2")

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

The SymPy translator (``smtlib2 -> sympy``) is experimental. Integer and real
formulas are translated directly; bit-vector symbols, constants, arithmetic,
and comparisons are approximated as unbounded integers, so overflow and
fixed-width wrapping are not modelled. Constructs it cannot translate raise
``NotImplementedError`` instead of producing a partial expression.
