.. _verifytools:

Verification Tools
==================

The ``aria.efmc.verifytools`` module provides internal supporting infrastructure
for verification workflows in EFMC, including intermediate representations,
Boogie AST and control-flow abstractions, and verification condition helpers.

Architecture
------------

.. code-block:: text

   aria/efmc/verifytools/
   ├── common/               # Common utilities and base AST
   │   ├── ast.py           # Base AstNode definition
   │   ├── parser.py        # Parser base classes (Parser, InfixExprParser)
   │   └── util.py          # Error handling and parsing utilities
   ├── boogie/              # Boogie program AST and CFG analysis
   │   ├── ast.py           # Boogie AST nodes (AstProgram, AstStmt, AstExpr)
   │   ├── bb.py            # Basic block graph extraction (get_bbs, BB)
   │   ├── eval.py          # Expression evaluation
   │   ├── grammar.py       # Boogie grammar rules
   │   ├── interp.py        # Abstract interpreter
   │   ├── paths.py         # Path enumeration
   │   ├── predicate_transformers.py  # Weakest precondition/postcondition
   │   ├── ssa.py           # SSA conversion helpers
   │   ├── analysis.py      # Flow analysis
   │   ├── inv_networks.py  # Invariant networks
   │   └── z3_embed.py      # Z3 SMT-LIB2 embedding helpers
   └── tools/               # Verification script tooling
       ├── vc_check.py      # Verification condition checking (tryAndVerifyLvl)
       ├── desugar.py       # Desugaring utilities for Boogie
       ├── tryAndVerify.py  # Try-and-verify CLI driver
       ├── conversions.py   # Format conversion helpers
       ├── boogie_loops.py  # Loop extraction from CFGs
       └── levels.py        # Level set management

Common AST and Utilities
------------------------

Base class for structural AST nodes:

.. code-block:: python

   from aria.efmc.verifytools.common.ast import AstNode

   class CustomNode(AstNode):
       def __init__(self, left, right):
           super().__init__(left, right)

Boogie AST and Basic Blocks
---------------------------

Parsing Boogie programs and extracting basic blocks:

.. code-block:: python

   from aria.efmc.verifytools.boogie.ast import parseAst, parseExprAst
   from aria.efmc.verifytools.boogie.bb import get_bbs

   # Parse Boogie expression AST
   expr = parseExprAst("x + 1 > 0")

   # Parse program AST from source
   # program = parseAst(source_code)
   # bbs, order = get_bbs(implementation)

Loop and Invariant Verification
-------------------------------

The ``tools`` package provides verification condition checking over extracted loops:

.. code-block:: python

   from aria.efmc.verifytools.tools.levels import loadBoogieLvlSet
   from aria.efmc.verifytools.tools.vc_check import tryAndVerifyLvl

   # Load benchmark levels
   # lvl_set_name, lvls = loadBoogieLvlSet("benchmark_set")

API Reference
-------------

.. automodule:: aria.efmc.verifytools.common.ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.efmc.verifytools.common.parser
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.efmc.verifytools.common.util
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.efmc.verifytools.boogie.ast
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.efmc.verifytools.boogie.bb
   :members:
   :undoc-members:
   :show-inheritance:
