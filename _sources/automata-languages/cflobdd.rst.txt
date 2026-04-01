CFL OBDD (Context-Free Language Ordered Binary Decision Diagrams)
================================================================

The ``aria.cflobdd`` module implements **CFLOBVDD** (Context-Free Language Ordered Bitvector Decision Diagrams), a generalization of CFLOBDDs for efficient symbolic computation on bitvector formulas.

This module is adapted from the Selfie Project (University of Salzburg).

.. contents:: Table of Contents
   :local:
   :depth: 2

Directory Structure
----------------

```
cflobdd/
├── __init__.py                    (0 lines - empty)
├── cflobvdd.py                    (1992 lines - main CFLOBVDD implementation)
├── bvdd.py                        (1012 lines - BVDD base classes)
├── btor2.py                       (1938 lines - BTOR2 parser/serializer)
├── z3interface.py                (438 lines - Z3 solver interface)
└── bitwuzlainterface.py          (514 lines - Bitwuzla solver interface)
```

**No subdirectories** - flat module structure.

Key Concepts
------------

### What are CFLOBVDDs?

**CFLOBVDDs generalize CFLOBDDs in two ways:**

1. **Multi-node BVDDs**: CFLOBVDDs are CFLOBDDs over **multi-node bitvector decision diagrams (BVDDs)** rather than single-node BDDs
   - A single BVDD node maps an n-bit bitvector to no more than 2^n different values (vs just 2 values for standard BDDs)
   - A tree of BVDD nodes of depth f maps a bitvector of size n*2^f bits to no more than 2^(n*2^f) values

2. **Minimization and Reordering**:
   - CFLOBVDDs are kept minimal with respect to all recursively pairwise reorderings
   - Reorderings are explored from root nodes down to a configurable "swap level"
   - Support for "fork level" (downsampling) and "swap level" (upsampling)

Key Components
--------------

### 1. BVDD (Bitvector Decision Diagrams) - ``bvdd.py``

**Core Hierarchy:**

* **BVDD_Node** - Base class for BVDD nodes (38 lines)
* **SBDD_i2o** - Single-byte decision diagram with naive input-to-output mapping (407 lines)
* **SBDD_s2o** - Single-byte decision diagram with set-to-output mapping (528 lines)
* **SBDD_o2s** - Single-byte decision diagram with output-to-set mapping (662 lines)
* **BVDD_uncached** - Uncached BVDD operations (773 lines)
* **BVDD_cached** - Cached BVDD operations (863 lines)
* **BVDD** - Main BVDD class (1011 lines)

**Key Features:**

* Works with 8-bit bitvectors (n=8)
* Uses 2^8-bit unsigned integers to represent sets of bitvector values
* Theta(2^n)-time set intersection via bitwise conjunction
* Extensive caching for performance
* Support for unary, binary, and ternary operations

**Performance Characteristics:**

* Set operations: O(2^n) for n-bit bitvectors
* Caching reduces repeated computations
* Thread-safe design for parallel usage

### 2. CFLOBVDD - ``cflobvdd.py``

**BV Grouping Hierarchy:**

* **BV_Grouping** - Base grouping class with caching infrastructure (46 lines)
* **BV_Dont_Care_Grouping** - Don't-care grouping (276 lines)
* **BV_Fork_Grouping** - Fork grouping for downsampling (382 lines)
* **BV_Internal_Grouping** - Internal grouping with swap operations (679 lines)
* **BV_No_Distinction_Proto** - No distinction prototype (1445 lines)
* **Collapsed_Classes** - Collapsed equivalence classes (1524 lines)
* **CFLOBVDD** - Main CFLOBVDD class (1568 lines)

**Key Operations:**

* Factory methods (class methods):
  - ``constant(level, swap_level, fork_level, output=0)``
  - ``byte_constant(level, swap_level, fork_level, number_of_input_bytes, output)``
  - ``false(level, swap_level, fork_level)``
  - ``true(level, swap_level, fork_level)``
  - ``projection(level, swap_level, fork_level, input_i, reorder=False)``
  - ``representative(grouping, outputs)``
  - ``print_profile()``
* Instance methods:
  - ``is_always_false()``
  - ``is_always_true()``
  - ``complement()``
  - ``unary_apply_and_reduce(op, number_of_output_bits)``
  - ``binary_apply_and_reduce(n2, op, number_of_output_bits)``
  - ``ternary_apply_and_reduce(n2, n3, op, number_of_output_bits)``
  - ``number_of_solutions(value)``

**Features:**

* Extensive caching with profile reporting
* Automatic minimization via reordering
* Support for fork/swap level configuration
* Thread-safe operations

### 3. BTOR2 Parser - ``btor2.py``

**Sort Classes:**

* **Sort** - Base sort class (188 lines)
* **Bitvector** - Bitvector sort (198 lines)
* **Bool** - Boolean sort (233 lines)
* **Bitvec** - Bitvec sort (242 lines)
* **Array** - Array sort (250 lines)

**Expression Classes:**

* **Expression** - Base expression (298 lines)
* **Constant**, **Zero**, **One**, **Constd**, **Const**, **Consth** - Constants (317-395 lines)
* **Variable** - Variables (421 lines)
* **Input**, **State** - Input and state variables (476-488 lines)
* **Ext**, **Slice** - Indexed operations (534-587 lines)
* **Unary**, **Binary** - Unary and binary operators (620-666 lines)
* **Implies**, **Comparison**, **Logical**, **Computation** - Specific binary ops (744-833 lines)
* **Concat**, **Read** - Bitvector operations (833-856 lines)
* **Ternary**, **Ite**, **Write** - Ternary operations (987-1115 lines)

**Transitional Classes:**

* **Transitional** - Base transition (1181 lines)
* **Init** - Initialization (1254 lines)
* **Next** - Next-state function (1303 lines)

**Property Classes:**

* **Property** - Base property (1344 lines)
* **Constraint** - Safety constraints (1360 lines)
* **Bad** - Bad state properties (1379 lines)

**Parser:**

* **Parser** - BTOR2 parser with tokenization (1409 lines)
* Supports full BTOR2 language with operators: sort, zero, one, const, constd, consth, input, state, init, next, ext, uext, slice, not, inc, dec, neg, implies, eq, neq, sgt, ugt, sgte, ugte, slt, ult, slte, ulte, and, or, xor, sll, srl, sra, add, sub, mul, sdiv, udiv, srem, urem, concat, read, ite, write, bad, constraint

### 4. Solver Interfaces

**Z3 Interface (``z3interface.py``):**

* **Z3** - Z3 wrapper (31 lines)
* **Bool**, **Bitvec**, **Array** - Z3 sort adapters (41-51 lines)
* **Expression** - Base expression with Z3 lambda generation (58 lines)
* **Constant**, **Constant_Array** - Constants (105-113 lines)
* **Input**, **State** - Variables (118-129 lines)
* **Ext**, **Slice** - Indexed operations (142-151 lines)
* **Unary**, **Implies**, **Comparison**, **Logical**, **Computation** - Operators (156-231 lines)
* **Concat**, **Read** - Operations (258-263 lines)
* **Ite**, **Write**, **Init**, **Next** - Higher-level operations (268-298 lines)
* **Property**, **Z3_Solver** - Property and solver (344-349 lines)

**Bitwuzla Interface (``bitwuzlainterface.py``):**

* Parallel structure to z3interface.py but for Bitwuzla solver
* Same class hierarchy with Bitwuzla-specific implementations
* **Bitwuzla_Solver** class with assertion, checking, and model generation

Usage Examples
--------------

### CFLOBVDD API
~~~~~~~~~~~~~~~

.. code-block:: python

   from aria.cflobdd import CFLOBVDD

   # Factory methods
   cflobvdd = CFLOBVDD.constant(level=3, swap_level=2, fork_level=1, output=0)
   cflobvdd_false = CFLOBVDD.false(level=3, swap_level=2, fork_level=1)
   cflobvdd_true = CFLOBVDD.true(level=3, swap_level=2, fork_level=1)
   cflobvdd_proj = CFLOBVDD.projection(level=3, swap_level=2, fork_level=1, input_i=2)

   # Instance methods
   cflobvdd.is_always_false()
   cflobvdd.is_always_true()
   complemented = cflobvdd.complement()
   result = cflobvdd.unary_apply_and_reduce(op, 8)
   result = cflobvdd.binary_apply_and_reduce(cflobvdd2, op, 8)
   result = cflobvdd.ternary_apply_and_reduce(cflobvdd2, cflobvdd3, op, 8)
   count = cflobvdd.number_of_solutions(42)

   # Print cache profile
   CFLOBVDD.print_profile()

### BVDD API
~~~~~~~~~~~~

.. code-block:: python

   from aria.cflobdd.bvdd import SBDD_i2o

   # Create constant
   sbdd = SBDD_i2o.constant(output_value=255)

   # Create projection
   sbdd = SBDD_i2o.projection(index=2, offset=0)

   # Operations
   result = sbdd.compute_unary(op='not')
   result = sbdd.compute_binary(op='and', bvdd2=sbdd2)
   result = sbdd.compute_ternary(op='ite', bvdd2=sbdd2, bvdd3=sbdd3)

   # Reduce
   reduced = sbdd.reduce_SBDD()
   reduced = sbdd.reduce_BVDD(index=0)

### BTOR2 Parser
~~~~~~~~~~~~~~

.. code-block:: python

   from aria.cflobdd.btor2 import Parser

   parser = Parser()
   model = parser.parse_btor2(modelfile='model.btor2', outputfile='output.txt')

   # Access parsed elements
   line = Line.get(nid=1)
   states = State.states  # Dictionary of all states
   inputs = Input.inputs  # Dictionary of all inputs
   nexts = Next.nexts  # Dictionary of next-state functions
   inits = Init.inits  # Dictionary of initializations
   constraints = Constraint.constraints  # Dictionary of constraints
   bads = Bad.bads  # Dictionary of bad state properties

### Solver Interface API
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Z3
   from aria.cflobdd.z3interface import Z3_Solver

   z3_solver = Z3_Solver(print_message=print, LAMBDAS=lambdas, UNROLL=5)
   z3_solver.push()
   z3_solver.assert_this(assertions, step=3)
   z3_solver.assert_not_this(assertions, step=4)
   z3_solver.prove()
   z3_solver.simplify()
   result = z3_solver.is_SAT(result)

   # Bitwuzla (similar interface)
   from aria.cflobdd.bitwuzlainterface import Bitwuzla_Solver

   bitwuzla_solver = Bitwuzla_Solver(print_message=print, LAMBDAS=lambdas, UNROLL=5)
   # ... same methods as Z3_Solver

Module Purpose Summary
--------------------

The cflobdd module provides:

1. **Efficient symbolic computation** using CFLOBVDD data structures for handling large bitvector formulas
2. **BTOR2 parsing** for reading bitvector transition systems from SMT-LIB compatible format
3. **Solver integration** with Z3 and Bitwuzla for verification and model checking
4. **Advanced decision diagram algorithms** with automatic minimization and reordering

This is likely used for:

* **Formal verification** - Model checking hardware/software systems
* **Symbolic execution** - Analyzing program behavior
* **Bitvector analysis** - Efficient reasoning over fixed-width bitvectors

Key Features
------------

1. **Minimal representations** - Automatic minimization via pairwise reordering
2. **Configurable levels** - Fork and swap level parameters for precision tuning
3. **Multi-node BVDDs** - More expressive than traditional BDDs
4. **Thread-safe** - Extensive use of threading locks for safe concurrent operations
5. **Comprehensive parsing** - Full BTOR2 language support
6. **Multiple solver backends** - Z3 and Bitwuzla integration
7. **Caching** - Extensive cache infrastructure with profiling support

Size Statistics
---------------

* Total Python lines: ~5,900
* CFLOBVDD implementation: ~1,992 lines
* BVDD implementation: ~1,012 lines
* BTOR2 parser: ~1,938 lines
* Z3 interface: ~438 lines
* Bitwuzla interface: ~514 lines

Source Attribution
----------------

Code adapted from Selfie Project (University of Salzburg) - `selfie.cs.uni-salzburg.at`

**License**: BSD License (see LICENSE file for details)

Notes
-----

* The module's ``__init__.py`` is empty - requires explicit imports
* No subdirectories - flat module structure
* Threading used extensively for thread-safe caching
* Both Z3 and Bitwuzla interfaces provided for flexibility
