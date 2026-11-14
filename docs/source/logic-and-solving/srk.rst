SRK Symbolic Reasoning Kit
==========================

The package ``aria.utils.srk`` is aria's Symbolic Reasoning Kit: a Python port of the
SRK framework for symbolic manipulation, verification, and automated reasoning.
It bundles expression syntax, algebraic domains, SMT backends, and termination
analyses under a single interface that interoperates with the rest of aria.

Overview
--------

* Build typed symbolic expressions and formulas with ``Context`` and helpers
* Integrate with SMT solvers through the generic ``SMTSolver`` API and the
  ``Z3Solver`` implementation
* Manipulate polynomials, polyhedra, and abstract domains for program analysis
* Encode transition systems, ranking functions, and fixpoint computations
* Parse, simplify, and export problems via SMT-LIB utilities

Subsystems
----------

**Syntax and Rewriting** ``aria.utils.srk.core.syntax``, ``aria.utils.srk.core.srkAst``, ``aria.utils.srk.logic.srkSimplify``
    Core classes for terms, formulas, and types. Provides builders such as
    ``mk_symbol``, ``mk_add``, ``mk_eq`` plus normalisation utilities like
    ``simplify_expression`` and CNF/NNF converters.

**SMT Integration** ``aria.utils.srk.logic.smt``, ``aria.utils.srk.logic.srkZ3``, ``aria.utils.srk.parser.srkSmtlib2``
    Translate SRK expressions to solver terms, manage solver scopes, and parse
    SMT-LIB 2 benchmarks. Includes helpers for models, unsat cores, and Z3
    interoperability.

**Abstract Interpretation and Algebra** ``aria.utils.srk.logic.abstract``, ``aria.utils.srk.linalg.interval``, ``aria.utils.srk.linalg.polyhedron``, ``aria.utils.srk.linalg.linear``, ``aria.utils.srk.algebra.polynomial``, ``aria.utils.srk.analysis.wedge``
    Implement numerical domains (intervals, affine relations, polyhedra),
    rational arithmetic (``qQ``/``zZ``), exponential polynomials, and operations
    needed for invariant generation and completeness proofs.

**Transition Systems and Termination** ``aria.utils.srk.lts.transition``, ``aria.utils.srk.lts.transitionFormula``, ``aria.utils.srk.lts.transitionSystem``, ``aria.utils.srk.termination.termination``, ``aria.utils.srk.analysis.loop``
    Model program control-flow, compose transitions, and search for ranking
    functions with LLRF/DTA/exp-based strategies.

**Utilities and Infrastructure** ``aria.utils.srk.structures.cache``, ``aria.utils.srk.structures.memo``, ``aria.utils.srk.misc.log``, ``aria.utils.srk.structures.sparseMap``, ``aria.utils.srk.parser.srkUtil``
    Support modules for performance profiling, memoisation, data structures, and
    combinatorial helpers used throughout the kit.

Getting Started
---------------

.. code-block:: python

   from aria.utils.srk import (
       Context, Type,
       mk_symbol, mk_const, mk_int, mk_add, mk_eq,
       simplify_expression, Z3Solver, SMTResult
   )

   ctx = Context()
   x = mk_symbol(ctx, "x", Type.INT)
   y = mk_symbol(ctx, "y", Type.INT)

   # mk_add takes a list of terms (or an explicit context and a list)
   # Both x and y occur, so both appear in the model.
   formula = mk_eq(mk_add([mk_const(x), mk_int(ctx, 3)]), mk_const(y))
   simplified = simplify_expression(formula, ctx)

   solver = Z3Solver(ctx)
   solver.add([simplified])

   if solver.check() == SMTResult.SAT:
       model = solver.get_model()
       print("Satisfiable:", model.get_value(x))
   else:
       print("Unsatisfiable or unknown")

Working with SMT-LIB
--------------------

``aria.utils.srk.parser.srkSmtlib2`` and
``aria.utils.srk.parser.srkSmtlib2Parse`` parse SMT-LIB 2 terms and models;
``aria.utils.srk.parser.srkSmtlib2Defs`` defines their AST, and
``aria.utils.srk.parser.srkParse`` handles SRK's own syntax (with a limited
SMT-LIB term subset). ``aria.utils.srk.logic.srkSimplify`` can normalise the
resulting formulas before they are handed to ``Z3Solver`` or exported back to
SMT-LIB. The combination allows rapid experimentation with new reasoning
algorithms while reusing SRK's infrastructure for parsing, simplification, and
solver interaction.

Further Reading
---------------

Explore the module docstrings and the extensive regression suite in
``aria/utils/srk/tests`` for concrete usage patterns covering algebraic manipulation,
quantifier handling, and program analysis workflows.
