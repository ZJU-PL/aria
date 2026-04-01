Predicate Abstraction and CEGAR
==============================


======================
Predicate Abstraction
======================
Abstraction aims at throwing away information about a system not needed to prove a given property. A safety invariant is an inductive invariant that implies the safety condition of a program.

An abstraction is a restricted language :math:`\mathcal{L}`. For example, in predicate abstraction, given a set of predicates :math:`\Pi`, we try to construct the strongest inductive invariant of a program expressible as a Boolean combination of :math:`\Pi`. That is, we are restricted to the language :math:`\mathcal{L}` of Boolean combinations of :math:`\Pi`.

**Definition (Predicate Abstraction Domain)**

Given a fixed finite set :math:`\Pi` of FO formulas with free variables from the program variables which captures the program semantics, a *predicate abstraction domain* consists of of lattice of propositional formulas over :math:`\Pi` ordered by implication.

Given a region formula :math:`\psi`, the *predicate abstraction* of :math:`\psi` with respect to the set of :math:`\Pi` of predicates is the smallest (in the implication ordering) region :math:`Abs(\psi, \Pi)` which contains :math:`\psi` and is representable as a Boolean combination of predicates from :math:`\Pi`:

.. math::

   Abs(\psi, \Pi) = \bigwedge \{  \phi \ | \ \phi \text{ is a Boolean formula over } \Pi \land \psi \Rightarrow \phi  \}


======================
Abstraction Refinement
======================

An abstraction refinement heuristic chooses a sequence of sublangauges :math:`\mathcal{L}_0 \subseteq \mathcal{L}_1, \ldots` from a broader language :math:`\mathcal{L}`. For example, in predicate abstraction with refinement, :math:`\mathcal{L}` is the set of quantifier-free first-order formulas, and :math:`\mathcal{L}_i` is characterized by a set of atomic predicates :math:`P_i`.

**Definition (Completeness relative to a language)**

An abstraction refinement heuristic is complete for language :math:`\mathcal{L}`, iff it always eventually chooses a sublanguage :math:`\mathcal{L}_i \subseteq \mathcal{L}` containing a safety invariant whenever :math:`\mathcal{L}` contains a safety invariant.


======================
References
======================

.. [BallR01] T. Ball and S. Rajamani. Automatically validating temporal safety properties of interfaces. In Proceedings of the 8th International SPIN Workshop on Model Checking of Software, pages 103-122, 2001.

.. [BallR02] T. Ball and S. Rajamani. The SLAM project: debugging system software via static analysis. In Proceedings of the 29th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, pages 1-3, 2002.
