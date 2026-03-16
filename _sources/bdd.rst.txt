BDD-based Symbolic Model Checking
================================

The BDD (Binary Decision Diagram) module implements symbolic model checking for Boolean programs using Z3's Boolean reasoning capabilities. This prover performs both forward and backward reachability analysis to verify safety properties.

======================
Overview
======================

BDD-based symbolic model checking is a technique that uses Binary Decision Diagrams to represent and manipulate Boolean functions efficiently. In EFMC, this is implemented using Z3's Boolean reasoning capabilities rather than traditional BDD libraries.


======================
Algorithm
======================

The BDD prover implements two main approaches:

**Forward Reachability Analysis:**
1. Start with the initial states
2. Iteratively compute the image (successor states) using the transition relation
3. Continue until a fixpoint is reached or the maximum number of iterations is exceeded
4. Check if the computed reachable states satisfy the safety property

**Backward Reachability Analysis:**
1. Start with the negation of the safety property (bad states)
2. Iteratively compute the pre-image (predecessor states) using the transition relation
3. Continue until a fixpoint is reached or the maximum number of iterations is exceeded
4. Check if the initial states are contained in the computed bad states


======================
Usage
======================

The BDD prover can be used as follows:

.. code-block:: python

    from efmc.engines.bdd import BDDProver
    from efmc.sts import TransitionSystem
    
    # Create transition system
    sts = TransitionSystem(...)
    
    # Create BDD prover with forward analysis
    prover = BDDProver(sts, use_forward=True, max_iterations=1000)
    
    # Solve the verification problem
    result = prover.solve()
    
    if result.is_safe:
        print("System is safe")
        print(f"Invariant: {result.invariant}")
    else:
        print("System is unsafe or unknown")

======================
Configuration Options
======================

- **use_forward**: Boolean flag to choose between forward (True) and backward (False) analysis
- **max_iterations**: Maximum number of iterations for fixpoint computation (default: 1000)

======================
Limitations
======================


- Currently uses Z3's Boolean reasoning rather than dedicated BDD libraries

======================
Future Improvements
======================

- Integration with dedicated BDD libraries (e.g., CUDD) for better performance

======================
References
======================


.. [1] E. M. Clarke, O. Grumberg, and D. A. Peled, "Model Checking," MIT Press, 1999. 