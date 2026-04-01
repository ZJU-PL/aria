Playing wth Quantifiers
========================

This page gives the conceptual background for quantified formulas. Technique-
specific pages such as :doc:`qe` and :doc:`qi` cover ARIA's current engines.

================
Quantifiers
================

Quantifiers are fundamental logical operators that express properties over sets of values:

- Universal Quantifier (∀): expresses that a property holds for all values
- Existential Quantifier (∃): expresses that a property holds for at least one value

=======================
Solving General Quantified Problems
=======================


The basic idea of quantifier instantiation is to replace a quantified formula with
a finite set of instances that are obtained by substituting concrete terms for the quantified variables.

Instantiation Techniques

1. **E-matching**:
2. **Model-based**: MBQI (model-based quantifier instantiation)
3. **Enumerative**:


Some useful tools:
- https://github.com/viperproject/axiom-profiler-2


=======================
Solving Exists-Forall Problems
=======================



=======================
Quantifier Elimination
=======================

Quantifier elimination refers to the process of eliminating the quantifiers from a formula by
constructing an equivalent quantifier-free formula.

We can apply Skolemization to remove the :math:`\exists` quantifiers. After that, it is sufficient to
instantiate the :math:`\forall` quantifiers for each constant.
For example, :math:`\forall x: T\ A` becomes :math:`A[c_1/x] \land A[c_2/x] \land \ldots`.
The result is equi-satisfiable and quantifier-free.


It is important to understand the differences between Skolemization and quantifier elimination:

- First, Skolemization does not preserve the formula's meaning (it preserves satisfiabilty, but not equivalence)
- Second, QE replaces a formula with an equivalent but quantifier-free formula
- Third, QE is only possible for specific theories, and is generally very expensive

=============
Related Work
=============

- Ge, Y.,  De Moura, L. (2009). Complete instantiation for quantified formulas in satisfiabiliby modulo theories. In Computer Aided Verification (pp. 306-320).
- de Moura, L., & Bjørner, N. (2007). Efficient E-matching for SMT solvers. In International Conference on Automated Deduction (pp. 183-198)
- Reasoning with Triggers, SMT'12
- Reynolds, A., Deters, M., Kuncak, V., Tinelli, C., & Barrett, C. (2015). Counterexample-guided quantifier instantiation for synthesis in SMT. In Computer Aided Verification (pp. 198-216).
- Counterexample-Guided Model Synthesis, TACAS'17
