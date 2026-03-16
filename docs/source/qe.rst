Quantifier Elimination (QE)
===========================

The Quantifier Elimination (QE) module uses quantifier elimination techniques to compute the strongest inductive invariant for a given transition system. This approach iteratively applies quantifier elimination to compute the least fixpoint of the strongest post-condition.

======================
Overview
======================

Quantifier Elimination is a technique that transforms quantified formulas into equivalent quantifier-free formulas. In the context of invariant generation, QE is used to compute the strongest inductive invariant by iteratively applying the strongest post-condition operator until a fixpoint is reached.



======================
Algorithm
======================

The QE prover implements the following algorithm:

1. **Initialization**: Start with the initial condition as the current invariant
2. **Iteration**: For each iteration:
   - Compute the strongest post-condition: ∃x. inv(x) ∧ trans(x,x')
   - Apply quantifier elimination to get a quantifier-free formula
   - Rename variables to get the next state formula
   - Take the disjunction with the current invariant
3. **Fixpoint Detection**: Check if the new invariant is equivalent to the previous one
4. **Safety Check**: Verify that the computed invariant implies the safety property

**Mathematical Formulation:**

The algorithm computes the least fixpoint of the function:

.. math::

   F(I) = init ∨ ∃x. (I(x) ∧ trans(x,x'))[x' ↦ x]

where `init` is the initial condition, `trans` is the transition relation, and `I` is the current invariant.

======================
Quantifier Elimination Tactics
======================

The QE prover uses multiple quantifier elimination tactics:

- **QE2**: Z3's advanced quantifier elimination tactic
- **QE**: Standard quantifier elimination tactic


======================
References
======================

.. [1] A. Tarski, "A decision method for elementary algebra and geometry," University of California Press, 1951.

.. [2] G. E. Collins, "Quantifier elimination for real closed fields by cylindrical algebraic decomposition," in ISSAC, 1975.

.. [3] L. de Moura and N. Bjørner, "Z3: An efficient SMT solver," in TACAS, 2008. 