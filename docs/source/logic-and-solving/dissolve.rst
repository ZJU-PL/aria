Dissolve (Distributed SAT)
===========================

``aria.bool.dissolve`` is a practical implementation of the Dissolve
distributed SAT solving framework. It uses asynchronous dilemma-rule
splits and clause sharing to parallelise SAT solving across workers.

Directory structure
-------------------

::

   aria/bool/dissolve/
   ├── dissolve.py         # Main Dissolve solver class
   ├── engine.py           # Worker engine logic
   ├── models.py           # DissolveConfig and DissolveResult data classes
   ├── scheduler.py        # Work distribution and scheduling
   └── ubtree.py           # UBTree-based clause store for subsumption checks

Key classes
------------

* ``Dissolve`` -- top-level solver; call ``solve(cnf)`` to run
* ``DissolveConfig`` -- configuration dataclass controlling solver behaviour:

  * ``k_split_vars`` -- number of variables used in dilemma splits
  * ``budget_strategy`` -- ``"constant"`` or ``"luby"`` sequence
  * ``distribution_strategy`` -- ``"dilemma"`` or ``"portfolio"``
  * ``clause_sharing`` -- enable / disable cross-worker clause exchange

* ``DissolveResult`` -- result container with satisfiability status, model,
  and runtime

Algorithm overview
------------------

1. **Split selection** -- pick ``k`` variables and create dilemma-rule
   sub-problems that partition the search space.
2. **Worker engines** -- each worker runs a PySAT-backed CDCL solver on its
   sub-problem with a configurable budget (constant or Luby).
3. **Clause sharing** -- learned clauses are shared across workers via a
   UBTree-based bucket store that supports subsumption checking.
4. **Vote aggregation** -- variable-picking heuristics aggregate votes from
   multiple workers to guide branching decisions.

Programmatic usage
------------------

.. code-block:: python

   from aria.bool.dissolve import Dissolve, DissolveConfig

   cfg = DissolveConfig(
       k_split_vars=5,
       budget_strategy="luby",
       distribution_strategy="dilemma",
   )
   solver = Dissolve(cfg)
   result = solver.solve(cnf_clauses)

   if result.is_sat:
       print("Model:", result.model)
