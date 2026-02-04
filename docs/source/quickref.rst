===============
Quick Reference
===============

This quick reference card provides a handy overview of common ARIA operations.

.. contents::
    :local:

------
Imports
-------

.. code-block:: python

    # Core imports
    from aria import srk
    from aria.smt import Solver
    from aria.smt.shortcuts import *

    # Specific modules
    from aria.counting import ModelCounter
    from aria.allsmt import AllSMT
    from aria.unsat_core import MUSExtractor
    from aria.backbone import BackboneComputer
    from aria.quant.qe import QuantifierEliminator
    from aria.abduction import Abductor
    from aria.sampling import SamplingEngine
    from aria.optimization import MaxSATSolver

-------
Variables
-------

.. code-block:: python

    # SRK variables
    x = srk.Integer("x")
    y = srk.Real("y")
    b = srk.Bool("b")
    bv = srk.BitVec("bv", 8)  # 8-bit vector

    # PySMT variables
    s = Solver()
    x = s.NewSymbol(INT, "x")
    y = s.NewSymbol(REAL, "y")
    b = s.NewSymbol(BOOL, "b")

-------
Operators
-------

Boolean Operators
^^^^^^^^^^^^^^^^^

.. code-block:: python

    And(a, b, c)           # Conjunction
    Or(a, b, c)            # Disjunction
    Not(a)                 # Negation
    Implies(a, b)         # Implication
    Ite(cond, then_, else_)  # If-then-else

Arithmetic Operators
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    +                      # Addition
    -                      # Subtraction
    *                      # Multiplication
    /                      # Division
    **                     # Power
    >, <, >=, <=, ==, !=  # Comparisons

Bit-Vector Operators
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    BVAdd(bv1, bv2)        # Addition
    BVSub(bv1, bv2)        # Subtraction
    BVMul(bv1, bv2)        # Multiplication
    BVAnd(bv1, bv2)        # Bitwise AND
    BVOr(bv1, bv2)         # Bitwise OR
    BVXor(bv1, bv2)        # Bitwise XOR
    BVLShr(bv, n)         # Logical shift right
    BVLShl(bv, n)         # Logical shift left

Array Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

    Store(arr, index, value)      # Update array
    Select(arr, index)           # Read from array

-------
Solver Operations
-------

.. code-block:: python

    s = Solver()

    # Add constraints
    s.add(x > 0)
    s.add(y > x)
    s.add(y < 100)

    # Check satisfiability
    if s.solve():
        # Get values
        print(s.get_value(x))
        print(s.get_value(y))

    # Get model as dictionary
    model = {x: s.get_value(x), y: s.get_value(y)}

    # Incremental solving
    s.push()
    s.add(x < 50)
    s.solve()
    s.pop()

-------
SRK Operations
-------

.. code-block:: python

    # Create formula
    formula = (x > 0) & (y > x) & (y < 100)

    # Check satisfiability
    result = srk.solve(formula)

    # Get model
    if result:
        model = srk.get_model(formula)
        print(model[x])
        print(model[y])

    # Simplify formula
    simplified = srk.simplify(formula)

    # Get all models
    for model in srk.get_models(formula, num_models=10):
        print(model)

-------
Advanced Operations
-------

Model Counting
^^^^^^^^^^^^^^

.. code-block:: python

    counter = ModelCounter(solver)
    count = counter.count()
    approx = counter.approx_count()

AllSMT
^^^^^^

.. code-block:: python

    all_smt = AllSMT(solver)
    for model in all_smt:
        print(model)

Unsat Core
^^^^^^^^^^

.. code-block:: python

    extractor = MUSExtractor(solver)
    core = extractor.extract()

Backbone
^^^^^^^^

.. code-block:: python

    computer = BackboneComputer(solver)
    backbone = computer.compute()

Quantifier Elimination
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    eliminator = QuantifierEliminator(solver)
    result = eliminator.eliminate(formula)

Abduction
^^^^^^^^^

.. code-block:: python

    abductor = Abductor(solver)
    explanations = abductor.abduce(background, observation)

Sampling
^^^^^^^^

.. code-block:: python

    sampler = SamplingEngine(solver)
    samples = sampler.sample(num_samples=10)

MaxSAT
^^^^^^

.. code-block:: python

    soft = SoftConstraint(x + y <= 80, weight=1)
    solver = MaxSATSolver(solver, [soft])
    result = solver.max_sat()

-------
Type Shorthands
-------

.. code-block:: python

    INT    # Integer type
    REAL   # Real number type
    BOOL   # Boolean type
    BV(n)  # Bit-vector of n bits
    FP(e, s)  # Floating-point (e exponent bits, s significand bits)
    ArrayType(dom, rng)  # Array type

-------
Common Patterns
-------

Check if constraint is satisfiable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    s = Solver()
    s.add(constraints...)
    if s.solve():
        print("Satisfiable!")
    else:
        print("Unsatisfiable")

Find any solution
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    s = Solver()
    s.add(constraints...)
    if s.solve():
        model = {v: s.get_value(v) for v in variables}
        print(model)

Find all solutions (limited)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    for i, model in enumerate(srk.get_models(formula)):
        if i >= 100: break
        print(model)

Optimize an objective
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    s = Solver()
    s.add(constraints...)
    soft = SoftConstraint(objective, weight=1)
    result = MaxSATSolver(s, [soft]).max_sat()

Debug unsatisfiable formulas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    s = Solver()
    s.add(constraints...)
    if not s.solve():
        core = MUSExtractor(s).extract()
        print(f"Unsatisfiable core: {core}")

-------
Environment Variables
-------

.. code-block:: bash

    ARIA_DEBUG=1              # Enable debug output
    ARIA_TIMEOUT=30           # Set default timeout (seconds)
    ARIA_SOLVER=z3           # Set default solver

-------
Error Handling
-------

.. code-block:: python

    from aria.smt import SolverError, SolverTimeoutError

    try:
        s = Solver()
        s.add(constraints)
        s.solve()
    except SolverTimeoutError:
        print("Solver timed out")
    except SolverError as e:
        print(f"Solver error: {e}")

-------
Performance Tips
-------

1. Use incremental solving with push/pop for related problems
2. Simplify formulas before solving complex constraints
3. Set appropriate timeouts to avoid hanging
4. Use the right theory (BV for hardware, INT for software, etc.)
5. Use parallel solving for large problems

-------
Getting Help
-------

- Documentation: https://zju-pl.github.io/aria
- GitHub: https://github.com/ZJU-PL/aria
- Issues: https://github.com/ZJU-PL/aria/issues
