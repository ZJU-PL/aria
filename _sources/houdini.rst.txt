Houdini
===================

The Houdini algorithm is a simple and efficient technique for inferring loop invariants.  It takes as input a set of predicates and outputs the maximal subset of these predicates that are inductive.

At a high level, the algorithm works as follows:
1. Start with a candidate invariant that is the conjunction of all predicates in the given set.
2. Check if this candidate invariant is inductive with respect to the program's transition relation.
3. If it is inductive, return it as the solution.
4. If not, identify the predicates that are violated in the inductiveness check.
5. Remove these violated predicates from the candidate set.
6. Repeat steps 2-5 with the reduced set of predicates until either:
   - The candidate invariant becomes inductive, or
   - All predicates have been eliminated (in which case the algorithm fails to find an inductive invariant from the given predicates).

The algorithm is guaranteed to terminate because each iteration removes at least one predicate from the candidate set, and there are finitely many predicates.

The key insight of Houdini is that if a conjunction of predicates is not inductive, then any conjunction containing the non-inductive predicates will also not be inductive. This allows Houdini to efficiently prune the search space.

The Houdini  algorithm is interesting for several reasons:
- It is an example of a "guess-and-check" style invariant inference. There are also many other technqeus for "guessing" such as SVM, neural networks, grammar-based enumeration.
- It can also be regarded as "template-based" invariant inference. We can regard the set of predicates as a speicial template, and the Houdini algorithm tries to find the most general template that is inductive. There are also other template-based invariant inference algorithms, such as those using the TCM domains (interval, octagon, etc.)
- It is can be used as a building block in more complex invariant inference algorithms. 

============
Houdini in EFMC
=============

The Houdini algorithm is implemented in the EFMC tool as a standalone invariant inference module.

=============
References
=============

.. [1] K. R. M. Leino and F. Logozzo, Houdini, an Annotation Assistant for ESC/Java. FME'01.

