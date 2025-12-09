
Abduction
=================================

==========================
Introduction to Abduction
==========================

Abduction is a type of logical inference that plays a crucial role in various fields, including artificial intelligence, philosophy, and scientific reasoning. It is distinct from induction and deduction, as it aims to generate hypotheses to explain observed facts, rather than proving or generalizing from given premises.

There are three major types of logical inference: induction, deduction, and abduction.

- In deduction, everything inferred is necessarily true,
while it is not the case with the remaining two types of inference.
- Induction tries to infer general rules based on individual instances.
- The aim of abduction is to produce additional hypotheses to explain observed facts.

Abduction has a wide spectrum of implicit or explicit applications
– in everyday life, in education, and in scientific reasoning, including in
building mathematical theories, or in software verification.
One definition of abduct is given below.



.. math::

   \text{Given a theory } T \text{ and a formula } G \text{ (the goal to be proved), such that } T \not\models G, \\
   \text{an explanation or } \textit{abduct} \text{ is a formula } A \text{ meeting the conditions: }

.. math::
   T, A \models G \quad \text{and} \quad T, A \not\models \bot.

It is clear that some abducts are not interesting, so there are often some additional restrictions given. There is no general agreement about such restrictions, but two types are most usual:

- **Syntactical restrictions**: abducts should be of a specific syntactical form.
- **Minimality restrictions**: for any other abduct :math:`A'`, if :math:`T, A \models A'`, then :math:`A \equiv A'`.

It is reasonable to ask that :math:`A` is not :math:`G`, as it is trivial. Some authors also add a stronger restriction that :math:`A \not\models G` (i.e., at least one axiom of :math:`T` has to be used to prove :math:`G`).

Various algorithms to produce different kinds of abducts have been developed.
In *abductive logic programming*, techniques for abductive reasoning are developed
in the context of logic programming. Rules are considered to be Horn clauses.

Some approaches are based on Robinson's resolution algorithm,
extended such that when no more clauses can be produced,
the atomic clauses are considered as potential abducts and
consistency is checked. There are also approaches developed for the context of SMT
solving, dealing with decidable theories like linear arithmetic.

In the context of geometry, some algebraic algorithms can generate additional
assumptions for the statement to be true. For example,
Wu's method can produce non-degeneracy conditions.
Algebraic methods can also be used to generate more general abducts.


======================
Abduction in ARIA
======================

Code Example
--------------

Here is an example of how to use ARIA to perform abductive reasoning:

.. code-block:: python

   from aria import abduction (TO FIX)

   # Define the theory and goal
   theory = ["p(X) :- q(X)", "q(a)"]
   goal = "p(b)"

   # Perform abduction
   abducts = abduction.generate_abducts(theory, goal)

   # Print the abducts
   for abduct in abducts:
       print(abduct)


=============
References
=============

- Kakas, A. C., Kowalski, R. A., & Toni, F. (1992). Abductive logic programming. Journal of logic and computation, 2(6), 719-770.

- Eiter, T., & Gottlob, G. (1995). The complexity of logic-based abduction. Journal of the ACM, 42(1), 3-42.

- Poole, D. (1993). Probabilistic Horn abduction and Bayesian networks. Artificial intelligence, 64(1), 81-129.
