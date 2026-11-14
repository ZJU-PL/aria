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

   from aria.efmc.engines.danger import DangerInvariantProver
   from aria.efmc.sts import TransitionSystem
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
- ``require_rank_positive``: Whether to require ranking function to be positive (bool, default: True). For unsigned bit-vector ranking functions the positivity conditions are skipped, since the checker only uses the unsigned strict-order comparison there.
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
-------------------------

Consider a 4-bit counter that starts at 0 and increments on every step, so it
wraps around after 15. The safety property to refute is ``x < 2``.

.. code-block:: python

   import z3
   from aria.efmc.engines.danger import DangerInvariantProver
   from aria.efmc.sts import TransitionSystem

   x = z3.BitVec("x", 4)
   x_p = z3.BitVec("x_p", 4)

   sts = TransitionSystem(
       variables=[x],
       prime_variables=[x_p],
       init=(x == 0),
       trans=(x_p == x + 1),
       post=(z3.ULT(x, z3.BitVecVal(2, 4))),
   )

   # Danger invariant: 0 <= x <= 3 (reachable from the initial state)
   danger_inv = z3.ULE(x, z3.BitVecVal(3, 4))
   # Ranking function: 4 - x
   ranking_func = z3.BitVecVal(4, 4) - x
   # Guard: loop continues while x < 3
   guard = z3.ULT(x, z3.BitVecVal(3, 4))

   prover = DangerInvariantProver(sts)
   result = prover.verify(danger_inv, ranking_func, guard=guard)

   # result.is_safe is False (and result.is_unsafe is True): the witness holds
   print(result.is_unsafe)

Use ``z3.BitVecVal`` for numeric constants in bit-vector formulas. Plain
Python integers larger than the bit width silently wrap (for an 8-bit vector,
``256`` becomes ``0``), which changes the meaning of guards and ranking
functions.

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

