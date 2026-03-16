Ranking Function Templates for Termination Verification
======================================================

This document describes the ranking function templates implemented for verifying termination of bit-vector programs in EFMC.

Overview
--------

Ranking functions are a fundamental technique for proving program termination. A ranking function is a function R that maps program states to a well-ordered domain (typically natural numbers) such that:

1. **Non-negativity**: R(x) ≥ 0 for all reachable states x
2. **Decrease**: For any transition from state x to state x', if the loop guard holds, then R(x) > R(x')
3. **Well-foundedness**: The range of R is bounded below

For bit-vector programs, the well-foundedness property is automatically satisfied due to the finite domain of bit-vectors.

Implemented Templates
---------------------

1. Linear Ranking Functions (``BitVecLinearRankingTemplate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear ranking functions have the form::

    R(x₁, x₂, ..., xₙ) = a₁*x₁ + a₂*x₂ + ... + aₙ*xₙ + c

where ``aᵢ`` and ``c`` are coefficients to be synthesized.

**Use case**: Simple loops with linear decreases, such as countdown loops.

**Example**::

    while (x > 0) {
        x = x - 1;
    }

Ranking function: ``R(x) = x``

2. Lexicographic Ranking Functions (``BitVecLexicographicRankingTemplate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lexicographic ranking functions use tuples ordered lexicographically::

    R(x) = (R₁(x), R₂(x), ..., Rₖ(x))

where each ``Rᵢ`` is a linear function. The tuple ``(a₁, a₂, ..., aₖ)`` is greater than ``(b₁, b₂, ..., bₖ)`` if there exists an index ``i`` such that:

- ``aⱼ = bⱼ`` for all ``j < i``
- ``aᵢ > bᵢ``

**Use case**: Nested loops or loops with multiple phases.

**Example**::

    while (x > 0) {
        y = x;
        while (y > 0) {
            y = y - 1;
        }
        x = x - 1;
    }

Ranking function: ``R(x, y) = (x, y)``

3. Conditional Ranking Functions (``BitVecConditionalRankingTemplate``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional ranking functions use if-then-else expressions::

    R(x) = if C₁(x) then R₁(x) else if C₂(x) then R₂(x) else ... else Rₙ(x)

where ``Cᵢ`` are conditions and ``Rᵢ`` are linear ranking functions.

**Use case**: Loops with different behaviors based on conditions.

**Example**::

    while (x > 0) {
        if (x % 2 == 0) {
            x = x / 2;
        } else {
            x = x - 1;
        }
    }

Ranking function: ``R(x) = if (x % 2 == 0) then 2*x else x``

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from efmc.sts import TransitionSystem
    from efmc.engines.ef.termination_prover import TerminationProver

    # Create your transition system
    sts = TransitionSystem(...)

    # Create termination prover
    prover = TerminationProver(sts)

    # Set ranking template
    prover.set_ranking_template("bv_linear_ranking")

    # Prove termination
    result = prover.prove_termination(timeout=30)

    if result.result:
        print("Termination proven!")
        ranking_func = prover.get_ranking_function()
        print(f"Ranking function: {ranking_func}")

Convenience Function
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from efmc.engines.ef.termination_prover import prove_termination_with_ranking_functions

    # Try multiple templates automatically
    success, ranking_func, template_used = prove_termination_with_ranking_functions(
        sts, 
        template_names=["bv_linear_ranking", "bv_lexicographic_ranking"],
        timeout=30
    )

    if success:
        print(f"Termination proven using {template_used}")
        print(f"Ranking function: {ranking_func}")

Template-Specific Options
~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Ranking Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    prover.set_ranking_template("bv_linear_ranking")

Lexicographic Ranking Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    prover.set_ranking_template("bv_lexicographic_ranking", num_components=3)

Conditional Ranking Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    prover.set_ranking_template("bv_conditional_ranking", num_branches=3)

Implementation Details
----------------------

Verification Condition Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The termination prover generates verification conditions that ensure:

1. **Non-negativity**: ``guard(x) ⇒ R(x) ≥ 0``
2. **Decrease**: ``guard(x) ∧ trans(x, x') ⇒ R(x) > R(x')``
3. **Initial state**: ``init(x) ⇒ R(x) ≥ 0`` (if applicable)

Template Constraints
~~~~~~~~~~~~~~~~~~~~

Each template adds specific constraints:

- **Linear**: Coefficients are bit-vector variables
- **Lexicographic**: Multiple sets of coefficients with lexicographic ordering constraints
- **Conditional**: Boolean condition variables and multiple coefficient sets

Bit-Vector Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Signedness**: Templates handle both signed and unsigned bit-vectors
- **Bit-width compatibility**: Automatic extension/truncation when needed
- **Overflow**: Bit-vector arithmetic naturally handles overflow

Limitations and Future Work
---------------------------

Current Limitations
~~~~~~~~~~~~~~~~~~~

1. **Template expressiveness**: Limited to linear combinations of variables
2. **Condition synthesis**: Conditional templates use simple boolean flags
3. **Nested loops**: Limited support for complex nested structures

Future Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Non-linear ranking functions**: Polynomial and other non-linear forms
2. **Automatic condition synthesis**: Learn conditions from program structure
3. **Multi-phase ranking functions**: Support for programs with distinct phases
4. **Ranking function composition**: Combine multiple ranking functions

Examples
--------

See ``examples/termination_example.py`` for complete working examples demonstrating all three template types.

References
----------

1. Bradley, A. R., Manna, Z., & Sipma, H. B. (2005). Termination of polynomial programs. In VMCAI.
2. Cook, B., Podelski, A., & Rybalchenko, A. (2006). Termination proofs for systems code. In PLDI.
3. Cousot, P. (2005). Proving program invariance and termination by parametric abstraction, Lagrangian relaxation and semidefinite programming. In VMCAI.
4. Heizmann, M., Hoenicke, J., & Podelski, A. (2013). Software model checking for people who love automata. In CAV. 