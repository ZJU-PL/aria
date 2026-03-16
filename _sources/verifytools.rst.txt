.. _verifytools:

Verification Tools
==================

The ``verifytools`` module provides supporting infrastructure for verification, including intermediate representations, verification condition generation, and integration with external tools.

Architecture
------------

.. code-block::

   efmc/verifytools/
   ├── common/               # Common utilities and AST
   │   ├── ast.py           # Abstract syntax tree definitions
   │   ├── parser.py        # Common parser utilities
   │   └── util.py          # Utility functions
   ├── boogie/              # Boogie-specific tooling
   │   ├── ast.py           # Boogie AST
   │   ├── bb.py            # Basic block analysis
   │   ├── eval.py          # Expression evaluation
   │   ├── grammar.py       # Boogie grammar
   │   ├── interp.py        # Interpreter
   │   ├── paths.py         # Path analysis
   │   ├── predicate_transformers.py  # Predicate transformers
   │   ├── ssa.py           # Static single assignment
   │   ├── analysis.py      # Static analysis
   │   ├── inv_networks.py  # Invariant networks
   │   └── z3_embed.py      # Z3 embedding
   ├── tools/               # Verification tools
   │   ├── vc_check.py      # Verification condition checking
   │   ├── desugar.py       # Desugaring utilities
   │   ├── tryAndVerify.py  # Try-and-verify strategy
   │   ├── conversions.py   # Format conversions
   │   ├── boogie_loops.py  # Loop extraction
   │   ├── levels.py        # Level management
   │   ├── run_ice.py       # ICE-based generation
   │   ├── run_invgen.py    # Invariant generation runner
   │   └── run_daikon.py    # Daikon integration
   ├── daikon/              # Daikon invariant detector
   │   ├── inv_ast.py       # Invariant AST
   │   └── inv_grammar.py   # Invariant grammar
   ├── cpachecker/          # CPAchecker integration
   │   └── dummy.h          # Stub header
   └── invgen/              # Invariant generation utilities

Common Utilities
----------------

The ``common`` subpackage provides shared utilities:

.. code-block:: python

   from efmc.verifytools.common.ast import Expr, Stmt
   from efmc.verifytools.common.parser import parse_formula
   from efmc.verifytools.common.util import simplify

Common AST
~~~~~~~~~~

Base classes for abstract syntax trees:

.. code-block:: python

   from efmc.verifytools.common.ast import (
       Expr,      # Base expression class
       Stmt,      # Base statement class
       Var,       # Variable reference
       Const,     # Constant
       BinOp,     # Binary operation
       UnOp,      # Unary operation
   )

   # Create expressions
   x = Var("x")
   expr = BinOp("+", x, Const(1))

Common Parser
~~~~~~~~~~~~~

Utility functions for parsing formulas:

.. code-block:: python

   from efmc.verifytools.common.parser import parse_formula

   # Parse from string
   formula = parse_formula("x + y > 0")

Boogie Frontend Tools
---------------------

The ``boogie`` subpackage provides tools for working with Boogie programs:

AST and Interpretation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.boogie.ast import BoogieProgram, Procedure
   from efmc.verifytools.boogie.interp import BoogieInterpreter

   # Parse Boogie program
   program = BoogieProgram.from_file("program.bpl")

   # Run interpreter
   interp = BoogieInterpreter(program)
   result = interp.run_procedure("main")

Basic Block Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.boogie.bb import BasicBlockAnalyzer

   analyzer = BasicBlockAnalyzer(program)
   blocks = analyzer.get_basic_blocks("procedure_name")

Static Single Assignment (SSA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.boogie.ssa import to_ssa

   ssa_program = to_ssa(program)

Predicate Transformers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.boogie.predicate_transformers import (
       wp,  # Weakest precondition
       sp,  # Strongest postcondition
   )

   # Compute weakest precondition
   wp_expr = wp(statement, postcondition)

Z3 Integration
~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.boogie.z3_embed import BoogieToZ3

   converter = BoogieToZ3()
   z3_expr = converter.convert(boogie_expr)

Verification Condition Generation
---------------------------------

The ``tools`` subpackage provides verification condition generation:

Verification Condition Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.tools.vc_check import check_vc

   result = check_vc(vc_formula, solver="z3")

Try-and-Verify Strategy
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.tools.tryAndVerify import TryAndVerify

   strategy = TryAndVerify(max_iterations=100)
   result = strategy.verify(transition_system)

Desugaring
~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.tools.desugar import desugar_program

   simplified = desugar_program(program)

Format Conversions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from efmc.verifytools.tools.conversions import (
       chc_to_smtlib,
       smtlib_to_chc,
   )

Daikon Integration
------------------

The ``daikon`` subpackage provides integration with the Daikon invariant detector:

.. code-block:: python

   from efmc.verifytools.daikon.inv_grammar import InvGrammar
   from efmc.verifytools.daikon.inv_ast import Invariant

   # Create invariant grammar
   grammar = InvGrammar()

   # Detect invariants from traces
   invariants = grammar.detect(traces)

CPAchecker Integration
----------------------

The ``cpachecker`` subpackage provides integration with CPAchecker:

.. code-block:: python

   from efmc.verifytools.cpachecker import run_cpachecker

   result = run_cpachecker(
       program="program.c",
       spec="safety.prp",
       timeout=300
   )

ICE-based Invariant Generation
------------------------------

The ICE (Counterexample-Driven, Consistency, Efficiency) learning strategy:

.. code-block:: python

   from efmc.verifytools.tools.run_ice import ICEGenerator

   generator = ICEGenerator()
   invariants = generator.learn(
       transition_system,
       max_samples=1000
   )

API Reference
-------------

.. automodule:: efmc.verifytools.common.ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.common.parser
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.common.util
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.boogie.ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.boogie.interp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.tools.vc_check
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.daikon.inv_ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.verifytools.daikon.inv_grammar
   :members:
   :undoc-members:
   :show-inheritance:
