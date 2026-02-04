========
Tutorial
========

This tutorial provides a comprehensive introduction to ARIA, a toolkit for automated reasoning and constraint solving. We'll cover basic concepts, practical examples, and advanced features.

.. contents::
    :local:
    :depth: 2

------------
Getting Started
------------

Installation
============

First, install ARIA using pip:

.. code-block:: bash

    pip install aria

Or for the latest development version:

.. code-block:: bash

    pip install git+https://github.com/ZJU-PL/aria.git

For local development:

.. code-block:: bash

    git clone https://github.com/ZJU-PL/aria
    cd aria
    pip install -e .

Basic Usage
===========

ARIA provides a high-level interface for creating and solving logical formulas. Here's a simple example:

.. code-block:: python

    from aria import srk

    # Create symbolic variables
    x = srk.Integer("x")
    y = srk.Integer("y")

    # Create a formula
    formula = (x > 0) & (y > x) & (y < 10)

    # Check satisfiability
    result = srk.solve(formula)
    print(f"Formula is satisfiable: {result}")

    # Get a model (concrete assignment)
    if result:
        model = srk.get_model(formula)
        print(f"Example solution: x={model[x]}, y={model[y]}")

Creating Formulas
=================

ARIA provides several ways to create formulas depending on your needs.

Using SRK (Symbolic Reasoning Kit)
----------------------------------

The SRK module is the core of ARIA for symbolic reasoning:

.. code-block:: python

    from aria.srk import Integer, Real, Bool, And, Or, Implies

    # Integer variables
    n = Integer("n")

    # Real (floating-point) variables
    x = Real("x")
    y = Real("y")

    # Boolean variables
    p = Bool("p")
    q = Bool("q")

    # Create formulas
    f1 = n > 0  # n > 0
    f2 = (x + y) == 10  # x + y = 10
    f3 = And(p, Or(q, f1))  # p AND (q OR n > 0)
    f4 = Implies(p, f2)  # p IMPLIES (x + y = 10)

Using PySMT Integration
------------------------

ARIA is built on PySMT, giving you access to its full API:

.. code-block:: python

    from aria.smt import Solver, REAL, INT, BOOL
    from pysmt.typing import Function, FunctionType

    # Create a solver
    s = Solver()

    # Create variables with types
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    # Add constraints
    s.add(x > 0)
    s.add(y > x)
    s.add(y < 100)

    # Check satisfiability
    if s.solve():
        print(f"x = {s.get_value(x)}")
        print(f"y = {s.get_value(y)}")

Working with Different Theories
===============================

Bit-Vectors
-----------

Bit-vectors represent fixed-width integers common in hardware verification and cryptography:

.. code-block:: python

    from aria.srk import BitVec, BVAdd, BVSub, BVAnd, BVLShr

    # Create 8-bit and 32-bit variables
    byte = BitVec("byte", 8)
    word = BitVec("word", 32)

    # Bit-vector operations
    doubled = BVAdd(byte, byte)
    masked = BVAnd(byte, 0x0F)  # Get lower 4 bits

    # Create a formula
    formula = (byte > 10) & (doubled < 100) & (BVLShr(byte, 2) == 5)

    # Solve
    result = srk.solve(formula)
    if result:
        model = srk.get_model(formula)
        print(f"byte = {model[byte]}")

Floating-Point Numbers
----------------------

ARIA supports IEEE 754 floating-point arithmetic:

.. code-block:: python

    from aria.smt.fp import FPSort, FP
    from aria.smt import Solver

    s = Solver()

    # Create floating-point variables (32-bit float)
    x = s.NewSymbol(FPSort(5, 11), "x")  # 5 exponent bits, 11 significand bits
    y = s.NewSymbol(FPSort(5, 11), "y")

    # Add constraints
    s.add(x > 0)
    s.add(y > x)
    s.add(x < 10)

    if s.solve():
        print(f"x = {s.get_value(x)}")
        print(f"y = {s.get_value(y)}")

Arrays
------

Arrays represent mappings from indices to values:

.. code-block:: python

    from aria.smt import Solver, ArrayType
    from pysmt.typing import INT

    s = Solver()

    # Create array type: Array[10] of integers
    arr_type = ArrayType(INT, INT)

    # Create array variable
    arr = s.NewSymbol(arr_type, "arr")

    # Access and modify arrays
    # arr[i] reads element at index i
    # Store(arr, i, v) creates new array with arr[i] = v

    from pysmt.operators import PLUS
    from aria.smt.shortcuts import Plus, Store, Ite

    i = s.NewSymbol(INT, "i")
    j = s.NewSymbol(INT, "j")

    s.add(i >= 0)
    s.add(i < 10)
    s.add(j >= 0)
    s.add(j < 10)
    s.add(Store(arr, i, 42)[j] != 42)

    if s.solve():
        print("Arrays can have different values at different indices")

Uninterpreted Functions
------------------------

Uninterpreted functions allow reasoning about functions without knowing their implementation:

.. code-block:: python

    from aria.smt import Solver
    from pysmt.typing import Function, FunctionType, INT

    s = Solver()

    # Create uninterpreted function type: f(int) -> int
    f_type = FunctionType(INT, [INT])
    f = s.NewSymbol(f_type, "f")

    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    # Add constraints about f
    s.add(f(x) > 0)
    s.add(f(y) > f(x))
    s.add(x < y)

    if s.solve():
        print("Found satisfying assignment with uninterpreted function f")

---------------
Practical Examples
---------------

Sudoku Solver
=============

Here's how to solve a Sudoku puzzle using ARIA:

.. code-block:: python

    from aria.smt import Solver, INT
    from pysmt.shortcuts import And, Or, Equals

    # Sudoku puzzle (0 for empty cells)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]

    s = Solver()

    # Create 81 variables for the grid
    cells = {}
    for i in range(9):
        for j in range(9):
            cells[(i, j)] = s.NewSymbol(INT, f"cell_{i}_{j}")

    # Constraint: Each cell is between 1 and 9
    for i in range(9):
        for j in range(9):
            s.add(cells[(i, j)] >= 1)
            s.add(cells[(i, j)] <= 9)

    # Constraint: Add given puzzle values
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                s.add(cells[(i, j)] == puzzle[i][j])

    # Constraint: Each row has unique values
    for i in range(9):
        for a in range(9):
            for b in range(a + 1, 9):
                s.add(cells[(i, a)] != cells[(i, b)])

    # Constraint: Each column has unique values
    for j in range(9):
        for a in range(9):
            for b in range(a + 1, 9):
                s.add(cells[(a, j)] != cells[(b, j)])

    # Constraint: Each 3x3 box has unique values
    for box_i in range(3):
        for box_j in range(3):
            cells_in_box = []
            for i in range(box_i * 3, box_i * 3 + 3):
                for j in range(box_j * 3, box_j * 3 + 3):
                    cells_in_box.append(cells[(i, j)])
            for a in range(9):
                for b in range(a + 1, 9):
                    s.add(cells_in_box[a] != cells_in_box[b])

    # Solve
    if s.solve():
        print("Solved Sudoku:")
        for i in range(9):
            row = []
            for j in range(9):
                row.append(str(s.get_value(cells[(i, j)])))
            print(" ".join(row))
    else:
        print("No solution exists")

Scheduling Problem
==================

Solve a simple task scheduling problem:

.. code-block:: python

    from aria.smt import Solver, INT
    from pysmt.shortcuts import Plus, LT, GT, And

    # Task scheduling: 4 tasks, each takes 1-3 time units
    # Task 0 must finish before task 1 starts
    # Task 1 must finish before task 2 starts
    # Task 2 must finish before task 3 starts
    # All tasks must complete by time 10

    n_tasks = 4
    max_time = 10

    s = Solver()

    # Start times for each task
    start = {i: s.NewSymbol(INT, f"start_{i}") for i in range(n_tasks)}
    # Durations for each task
    duration = {i: s.NewSymbol(INT, f"duration_{i}") for i in range(n_tasks)}

    # Each duration is 1-3
    for i in range(n_tasks):
        s.add(duration[i] >= 1)
        s.add(duration[i] <= 3)

    # All tasks start at or after time 0
    for i in range(n_tasks):
        s.add(start[i] >= 0)

    # Sequential scheduling (task i finishes before task i+1 starts)
    for i in range(n_tasks - 1):
        finish_i = start[i] + duration[i]
        s.add(start[i + 1] >= finish_i)

    # All tasks complete by max_time
    for i in range(n_tasks):
        s.add(start[i] + duration[i] <= max_time)

    if s.solve():
        print("Scheduling found:")
        for i in range(n_tasks):
            print(f"  Task {i}: start={s.get_value(start[i])}, "
                  f"duration={s.get_value(duration[i])}, "
                  f"finish={s.get_value(start[i] + duration[i])}")
    else:
        print("No valid schedule exists")

Cryptographic Constraint Solving
================================

Solve constraints from a simple block cipher:

.. code-block:: python

    from aria.smt import Solver, INT
    from pysmt.shortcuts import Equals, Xor, And

    # Simplified S-box constraint problem
    # Given: y = S(x) where S is a 4-bit S-box
    # Find x such that y = 0xA (1010 binary)

    s = Solver()

    # 4-bit input and output
    x = {i: s.NewSymbol(INT, f"x_{i}") for i in range(4)}
    y = {i: s.NewSymbol(INT, f"y_{i}") for i in range(4)}

    # x and y are bits (0 or 1)
    for i in range(4):
        s.add(x[i] >= 0)
        s.add(x[i] <= 1)
        s.add(y[i] >= 0)
        s.add(y[i] <= 1)

    # S-box definition: y = S(x)
    # S-box mapping (simplified):
    # 0000 -> 1111 (15)
    # 0001 -> 0111 (7)
    # 0010 -> 0011 (3)
    # 0011 -> 1100 (12)
    # ... add more as needed

    # Want y = 1010 (decimal 10)
    # This means: y3=1, y2=0, y1=1, y0=0
    s.add(y[0] == 0)  # y0 = 0
    s.add(y[1] == 1)  # y1 = 1
    s.add(y[2] == 0)  # y2 = 0
    s.add(y[3] == 1)  # y3 = 1

    # Add S-box constraints (simplified example)
    # y[0] = x[0] XOR x[1] XOR 1
    s.add(y[0] == x[0] + x[1] - 2 * x[0] * x[1])

    # y[1] = NOT x[0] AND x[2]
    s.add(y[1] == (1 - x[0]) * x[2])

    # y[2] = x[1] XOR x[3]
    s.add(y[2] == x[1] + x[3] - 2 * x[1] * x[3])

    # y[3] = x[0] OR x[2]
    s.add(y[3] == x[0] + x[2] - x[0] * x[2])

    if s.solve():
        x_val = "".join(str(s.get_value(x[i])) for i in range(3, -1, -1))
        y_val = "".join(str(s.get_value(y[i])) for i in range(3, -1, -1))
        print(f"Input x = {x_val} (decimal {int(x_val, 2)})")
        print(f"Output y = {y_val} (decimal {int(y_val, 2)})")
    else:
        print("No solution exists for this S-box configuration")

-----------------
Advanced Features
-----------------

Model Counting
===============

Count the number of satisfying assignments:

.. code-block:: python

    from aria.counting import ModelCounter

    from aria.smt import Solver, INT

    # Create a simple formula
    s = Solver()
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    s.add(x > 0)
    s.add(x < 5)
    s.add(y > 0)
    s.add(y < 5)
    s.add(y != x)

    # Count models
    counter = ModelCounter(s)
    count = counter.count()
    print(f"Number of solutions: {count}")

    # Get approximate count for large solution spaces
    approx = counter.approx_count()
    print(f"Approximate count: {approx}")

AllSMT (All Satisfying Models)
==============================

Enumerate all satisfying models:

.. code-block:: python

    from aria.allsmt import AllSMT

    from aria.smt import Solver, INT

    s = Solver()
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    s.add(x > 0)
    s.add(x < 4)
    s.add(y > 0)
    s.add(y < 4)

    # Enumerate all models
    all_smt = AllSMT(s)
    for i, model in enumerate(all_smt):
        if i >= 10:  # Limit output
            print(f"... and more models exist")
            break
        print(f"Model {i+1}: x={model[x]}, y={model[y]}")

Unsat Core Extraction
=====================

Find the minimal unsatisfiable subset of constraints:

.. code-block:: python

    from aria.unsat_core import MUSExtractor

    from aria.smt import Solver, INT

    s = Solver()

    # Create a satisfiable base formula
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    s.add_assertion(x > 0, name="x_positive")
    s.add_assertion(y > 0, name="y_positive")
    s.add_assertion(x + y < 10, name="sum_constraint")

    # Now add an unsatisfiable constraint
    s.add_assertion(x > 100, name="x_large")  # Contradicts x + y < 10

    # Extract unsat core
    extractor = MUSExtractor(s)
    core = extractor.extract()

    print(f"Unsatisfiable core: {core}")

Backbone Literals
=================

Find literals that are true in all models:

.. code-block:: python

    from aria.backbone import BackboneComputer

    from aria.smt import Solver, INT

    s = Solver()

    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    s.add(x > 0)
    s.add(x < 10)
    s.add(y > x)
    s.add(y < 5)

    # Compute backbone
    computer = BackboneComputer(s)
    backbone = computer.compute()

    print(f"Backbone literals: {backbone}")
    print(f"These literals are true in ALL satisfying assignments")

Quantifier Elimination
=====================

Eliminate quantifiers from formulas:

.. code-block:: python

    from aria.quant.qe import QuantifierEliminator

    from aria.smt import Solver, INT

    s = Solver()
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    # Formula: exists x. (x > 0 and x < y)
    # Eliminating x gives: y > 0

    from pysmt.typing import BOOL
    from pysmt.shortcuts import Plus, Equals, GT, LT, And, Exists

    formula = Exists([x], And(x > 0, x < y))

    eliminator = QuantifierEliminator(s)
    result = eliminator.eliminate(formula)

    print(f"Original: exists x. (x > 0 and x < y)")
    print(f"After QE: {result}")

Abductive Reasoning
===================

Find explanations for observations:

.. code-block:: python

    from aria.abduction import Abductor

    from aria.smt import Solver, INT
    from pysmt.shortcuts import And, GT, LT, Implies

    s = Solver()

    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    # Background knowledge: x > 0 and x < 10 implies y > 5
    background = And(x > 0, x < 10, Implies(And(x > 0, x < 10), y > 5))

    # Observation: y > 5
    observation = y > 5

    # Find explanations
    abductor = Abductor(s)
    explanations = abductor.abduce(background, observation)

    print(f"Observation: {observation}")
    print(f"Possible explanations:")
    for exp in explanations:
        print(f"  {exp}")

Sampling Solutions
===================

Generate diverse solutions:

.. code-block:: python

    from aria.sampling import SamplingEngine

    from aria.smt import Solver, INT

    s = Solver()
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    s.add(x > 0)
    s.add(x < 100)
    s.add(y > 0)
    s.add(y < 100)

    # Sample diverse solutions
    sampler = SamplingEngine(s)

    # Get 5 diverse samples
    samples = sampler.sample(num_samples=5)
    for i, model in enumerate(samples):
        print(f"Sample {i+1}: x={model[x]}, y={model[y]}")

Optimization (MaxSAT)
====================

Find the optimal solution:

.. code-block:: python

    from aria.optimization import MaxSATSolver, SoftConstraint

    from aria.smt import Solver, INT

    s = Solver()

    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(INT, "y")

    # Hard constraints
    s.add(x >= 0)
    s.add(y >= 0)
    s.add(x + y <= 100)

    # Soft constraints (optimization objectives)
    soft1 = SoftConstraint(x + y <= 80, weight=1)  # Prefer x + y <= 80
    soft2 = SoftConstraint(x >= y, weight=2)  # Strongly prefer x >= y

    solver = MaxSATSolver(s, [soft1, soft2])
    result = solver.max_sat()

    if result.success:
        print(f"Optimal x = {result.model[x]}")
        print(f"Optimal y = {result.model[y]}")
        print(f"Soft constraints satisfied: {result.num_soft_satisfied}")

--------------
Best Practices
--------------

Performance Tips
================

1. **Use incremental solving**: When solving multiple related problems, use incremental solving to share constraints between solves.

2. **Simplify formulas**: Use ARIA's simplification passes before solving complex formulas.

3. **Choose the right theory**: Use bit-vector solvers for hardware problems, integer solvers for software, etc.

4. **Set appropriate timeouts**: Always set timeouts for solver calls to avoid hanging.

5. **Use parallel solving**: For large problems, use ARIA's parallel solving capabilities.

Debugging Tips
==============

1. **Check satisfiability first**: Always check if a formula is satisfiable before trying to get a model.

2. **Use model-based debugging**: When a formula is unsatisfiable, use unsat core extraction to find the problematic constraints.

3. **Enable debug mode**: Set `ARIA_DEBUG=1` to get detailed solver output.

4. **Simplify incrementally**: Start with a simple formula and add constraints one at a time to identify issues.

Error Handling
==============

Always handle solver exceptions:

.. code-block:: python

    from aria.smt import Solver, SolverError, SolverTimeoutError

    s = Solver()
    x = s.NewSymbol(INT, "x")

    try:
        s.add(x > 0)
        if s.solve():
            print(f"x = {s.get_value(x)}")
    except SolverTimeoutError:
        print("Solver timed out - try simplifying the problem")
    except SolverError as e:
        print(f"Solver error: {e}")

------------
Next Steps
----------

Now that you've completed this tutorial, explore these topics:

- :doc:`smt` - Detailed SMT solving documentation
- :doc:`srk` - Symbolic reasoning kernel capabilities
- :doc:`quantifiers` - Quantifier handling and elimination
- :doc:`optimization` - MaxSAT and optimization solving
- :doc:`counting` - Model counting techniques
- :doc:`applications` - Real-world application examples

For more examples, see the ``examples/`` directory in the repository.
