K-induction
==============================


Formal Verification Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Model Checking**: Automatically explores all possible states of a system to verify properties.
- **Theorem Proving**: Involves manually or semi-automatically proving properties using logical reasoning.
- **Abstract Interpretation**: Analyzes programs by abstracting their behaviors to predict properties.

K-induction falls under the **model checking** umbrella, extending traditional inductive methods to verify more intricate properties.

Mathematical Induction Refresher
---------------------------------

Mathematical induction is a foundational proof technique used to establish the truth of an infinite sequence of propositions.

Components of Mathematical Induction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Base Case**: Verify the statement for an initial value, typically `n = 0` or `n = 1`.
2. **Inductive Step**: Assume the statement holds for `n = k` (inductive hypothesis) and prove it for `n = k + 1`.

Example: Sum of the First `n` Natural Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prove that for all natural numbers `n`, the sum of the first `n` natural numbers is `(n(n + 1))/2`.

**Proof:**

1. **Base Case (`n=1`)**:

   Sum = 1 = (1 × (1 + 1))/2 = 1 ✓

2. **Inductive Step**:

   - **Inductive Hypothesis**: Assume true for `n = k`:
     Sum = `(k(k + 1))/2`

   - **To Prove for `n = k + 1`**:

   .. math::

      \text{Sum} = \frac{k(k + 1)}{2} + (k + 1) = \frac{k(k + 1) + 2(k + 1)}{2} = \frac{(k + 2)(k + 1)}{2}

   Thus, the statement holds for `n = k + 1`.

By the principle of mathematical induction, the formula is valid for all natural numbers `n`.

Inductive Verification
-----------------------

Inductive verification adapts mathematical induction principles to verify properties of systems modeled as state transitions.

Basic Inductive Verification Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Base Case**: Ensure the property `P` holds in the initial state(s) of the system.
2. **Inductive Step**: Show that if `P` holds in an arbitrary state, it holds in all successor states.

If both steps are satisfied, `P` is an *invariant* of the system, meaning it remains true regardless of the system's evolution.

Formal Definition
~~~~~~~~~~~~~~~~~~

Given a transition system with states and transitions, to verify a property `P`:

.. math::

   \text{Initial States}: \{ s_0 \ | \ P(s_0) \}
   \text{Transition Relation}: s \rightarrow s'
   \text{Property}: \forall s, s', \text{ if } P(s) \text{ then } P(s')

Challenges in Inductive Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Complex State Spaces**: Systems with large or infinite states make inductive proofs non-trivial.
- **Choosing Inductive Hypotheses**: The strength and correctness of the inductive step depend on the hypotheses used.

K-induction addresses some of these challenges by considering multiple states in the inductive step.


K-Induction Explained
-----------------------

**K-induction** extends basic inductive verification by considering `k` consecutive states in the inductive step, enhancing the power of the inductive hypothesis.

Motivation for K-Induction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Dependent Properties**: Some properties depend on multiple preceding states.
- **Enhanced Expressiveness**: K-induction can capture temporal behaviors that simple induction cannot.

Components of K-Induction
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Base Cases**:
   - Verify that the property `P` holds for the first `k` consecutive states starting from initial states.

2. **Inductive Step**:
   - **Inductive Hypothesis**: Assume `P` holds for `k` consecutive states.
   - **To Prove**: `P` holds for the next state `(k + 1)`.

Formal Definition
~~~~~~~~~~~~~~~~~~

Given a system with a transition relation :math:`\rightarrow` and a property :math:`P`, k-induction involves:

1. **Base Cases**:

   .. math::

      \text{Initial States}: P(s_0), P(s_1), \ldots, P(s_{k-1})

2. **Inductive Step**:

   .. math::

      \forall s_0, s_1, \ldots, s_k, \text{ if } P(s_0) \land P(s_1) \land \ldots \land P(s_{k-1}), \text{ then } P(s_k)

Choosing `k`
~~~~~~~~~~~~~~

- **Small Systems**: `k = 1` may suffice.
- **Complex Dependencies**: Larger values of `k` may be necessary to capture necessary dependencies.


======================
References
======================

- Clarke, E. M., Grumberg, O., & Peled, D. A. (1999). Model checking. MIT press.
- Donaldson, A. F., Haller, L., & Kroening, D. (2011). Strengthening induction-based verification using k-induction. In Formal Methods in Computer-Aided Design (FMCAD), 2011 (pp. 51-58). IEEE.
- Bradley, A. R., & Manna, Z. (2007). The Calculus of Computation: Decision Procedures with Applications to Verification. Springer Science & Business Media.
- Sheeran, M., Singh, S., & Stålmarck, G. (2000). Checking safety properties using induction and a SAT-solver. In International Conference on Formal Methods in Computer-Aided Design (pp. 108-125). Springer, Berlin, Heidelberg.