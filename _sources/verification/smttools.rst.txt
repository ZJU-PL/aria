SMT Tools
=========

``aria.efmc.smttools`` provides SMT-level utilities used across EFMC
engines: PySMT-enhanced solving, Craig interpolation, portfolio and
exists-forall (EFSMT) solving, and SyGuS-based function synthesis via CVC5.

Directory structure
-------------------

::

   aria/efmc/smttools/
   ├── pysmt_solver.py      # PySMTSolver and helpers
   ├── sygus_solver.py      # SyGuS encoding, CVC5 invocation, result parsing
   └── smt_exceptions.py    # SMTError / SmtlibError / SolverError hierarchy

PySMTSolver
------------

``PySMTSolver`` extends ``z3.Solver`` with PySMT-backed capabilities.

**Key methods:**

* ``convert(zf)`` -- static; converts a Z3 formula to PySMT, returning
  ``(pysmt_vars, pysmt_fml)``
* ``check_with_pysmt()`` -- satisfiability check via PySMT backend
* ``check_portfolio()`` -- parallel portfolio solving across MathSAT (×3),
  CVC4, and Yices
* ``all_smt(keys, bound=5)`` -- samples up to ``bound`` models by blocking
  partial assignments
* ``binary_interpolant(fml_a, fml_b, ...)`` -- Craig binary interpolation,
  result mapped back to Z3
* ``sequence_interpolant(formulas)`` -- sequence interpolation over a list of
  formulas
* ``efsmt(evars, uvars, z3fml, ...)`` -- solves ``∃x. ∀y. φ(x,y)`` via a
  CEGIS-style loop; supports ``maxloops``, ``timeout``, and configurable
  existential/universal solvers

**Helper function:**

* ``to_pysmt_vars(z3vars)`` -- converts a list of Z3 variables to PySMT
  ``FNode`` variables (supports Int, Real, BitVector, Boolean)

SyGuS Solver
-------------

The ``sygus_solver`` module encodes synthesis problems in the SyGuS format,
invokes CVC5, and maps results back to Z3 expressions.

**Key functions:**

* ``synthesize_function(func, constraints, variables, ...)`` -- top-level
  synthesis entry point; auto-detects logic (LIA, BV, String), uses shortcut
  implementations for common functions (``max``, ``min``, ``xor``, ``concat``,
  etc.), and falls back to CVC5
* ``build_sygus_cnt(funcs, cnts, variables, logic)`` -- encodes a Z3
  synthesis specification into SyGuS input format
* ``solve_sygus(sygus_problem, timeout)`` -- writes the problem to a temp file
  and invokes CVC5
* ``parse_sygus_solution(cvc5_output)`` -- parses CVC5's ``define-fun``
  output into a structured dictionary
* ``sygus_to_z3(func_name, func_def, variables)`` -- converts a parsed SyGuS
  function definition back to a Z3 expression

Exceptions
----------

* ``SMTError`` -- base exception for SMT-related errors
* ``SmtlibError`` -- SMT-LIB format errors
* ``SolverError`` -- solver invocation / runtime errors

Programmatic usage
------------------

**Interpolation:**

.. code-block:: python

   from aria.efmc.smttools.pysmt_solver import PySMTSolver
   import z3

   solver = PySMTSolver()
   a, b = z3.Bools("a b")
   itp = solver.binary_interpolant(z3.Implies(a, b), z3.Not(b))

**Portfolio solving:**

.. code-block:: python

   solver = PySMTSolver()
   solver.add(constraints)
   result = solver.check_portfolio()  # parallel across 5 backends

**SyGuS synthesis:**

.. code-block:: python

   from aria.efmc.smttools.sygus_solver import synthesize_function

   result = synthesize_function(
       func=uninterp_func,
       constraints=[phi1, phi2],
       variables=[x, y],
       timeout=30,
   )
