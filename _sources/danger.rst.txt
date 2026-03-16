Danger Invariants
=================

Overview
--------

Danger invariants provide a dual approach to traditional safety invariants. While safety invariants prove that a program is safe (no bad states are reachable), danger invariants prove that a program is unsafe (bad states are reachable). This is particularly useful for bug finding and unsafety verification.

A danger invariant D(x) together with a ranking function R(x) forms an unsafety witness that proves the existence of a reachable bad state.

Verification Conditions
-----------------------

For a danger invariant D(x) and ranking function R(x), the following conditions must hold:

1. **Base / Reachability:**
   
   .. math::
      
      \exists x. \text{Init}(x) \wedge D(x)
   
   The danger invariant must be reachable from an initial state.

2. **Progress:**
   
   .. math::
      
      \forall x. D(x) \wedge G(x) \wedge \text{Post}(x) \Rightarrow \exists x'. \text{Trans}(x,x') \wedge D(x') \wedge R(x') < R(x)
   
   From any state satisfying the danger invariant, there must exist a transition to another state in the danger invariant with a strictly decreasing ranking function.
   
   Additionally, the ranking function must be positive:
   
   .. math::
      
      \forall x. D(x) \Rightarrow R(x) > 0

3. **Exit-Violation** (when a guard G is provided):
   
   .. math::
      
      \forall x. D(x) \wedge \neg G(x) \Rightarrow \neg \text{Post}(x)
   
   When exiting the loop (guard is false), the safety property must be violated.

Result Interpretation
---------------------

The danger invariant prover returns a `VerificationResult` with the following semantics:

- **is_safe = True**: The provided witness is invalid (no bug proved)
- **is_safe = False**: The witness is valid (bug proved)
- **is_unknown = True**: The solver answered unknown, or the witness was rejected but no safety proof has been given

Usage
-----

The danger invariant prover is primarily used programmatically:

.. code-block:: python

   from efmc.engines.danger import DangerInvariantProver
   from efmc.sts import TransitionSystem
   import z3
   
   # Create transition system
   sts = TransitionSystem(...)
   
   # Create prover
   prover = DangerInvariantProver(sts)
   
   # Define danger invariant and ranking function
   x = sts.variables[0]
   danger_inv = x > 100  # Example danger invariant
   ranking_func = 200 - x  # Example ranking function
   guard = x < 200  # Optional loop guard
   
   # Verify the danger invariant
   result = prover.verify(
       inv=danger_inv,
       rank=ranking_func,
       guard=guard,
       require_rank_positive=True,
       solver_timeout_ms=60000
   )
   
   if not result.is_safe:
       print("Bug proved! Unsafety witness is valid.")
       print(f"Danger invariant: {danger_inv}")
       print(f"Ranking function: {ranking_func}")
   else:
       print("Witness rejected or unknown")

API Reference
-------------

**DangerInvariantProver.verify()**

Parameters:
- ``inv``: The danger invariant formula (z3.ExprRef)
- ``rank``: The ranking function formula (z3.ExprRef)
- ``guard``: Optional loop guard formula (z3.ExprRef, default: None)
- ``require_rank_positive``: Whether to require ranking function to be positive (bool, default: True)
- ``solver_timeout_ms``: Solver timeout in milliseconds (int, default: None)

Returns:
- ``VerificationResult`` with the verification outcome

Bit-Vector Support
------------------

The danger invariant prover supports both signed and unsigned bit-vector interpretations:

- For signed bit-vectors: Uses signed comparison (``>``)
- For unsigned bit-vectors: Uses unsigned comparison (``UGT``)

The signedness is determined from the transition system's configuration.

Example: Proving Unsafety
--------------------------

Consider a program that increments a counter until it overflows:

.. code-block:: python

   # Transition system where x starts at 0 and increments
   # Safety property: x < 256
   # We want to prove this is violated
   
   x = z3.BitVec('x', 8)
   x_p = z3.BitVec('x_p', 8)
   
   # Danger invariant: x >= 255 (close to overflow)
   danger_inv = z3.UGE(x, 255)
   
   # Ranking function: 256 - x (decreases as x increases)
   ranking_func = 256 - x
   
   # Guard: x < 256 (loop continues)
   guard = z3.ULT(x, 256)
   
   prover = DangerInvariantProver(sts)
   result = prover.verify(danger_inv, ranking_func, guard=guard)
   
   # If result.is_safe is False, we've proved the program is unsafe

Applications
------------

- **Bug Finding**: Prove that certain error states are reachable
- **Counterexample Generation**: Generate concrete paths to buggy states
- **Refinement**: Use danger invariants to guide abstraction refinement
- **Dual Verification**: Complement safety verification with unsafety proofs

Related Work
------------

- Heizmann, M., et al. "Software model checking for people who love automata." CAV 2013.
- Cook, B., et al. "Proving program termination." CACM 2011.
- Bradley, A. R., et al. "Termination of polynomial programs." VMCAI 2005.

