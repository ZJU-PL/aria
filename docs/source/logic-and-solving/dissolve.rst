Dissolve (Distributed SAT)
==========================

``aria.bool.dissolve`` is a practical implementation of the Dissolve
distributed SAT solving framework. It uses asynchronous dilemma-rule
splits and clause sharing to parallelise SAT solving across workers.

Directory structure
-------------------

::

   aria/bool/dissolve/
   ├── dissolve.py         # Main Dissolve solver class
   ├── engine.py           # Dilemma-rule worker engine helpers (used by the demo)
   ├── models.py           # DissolveConfig and DissolveResult data classes
   ├── scheduler.py        # Scheduling helpers (used by the demo)
   └── ubtree.py           # UBTree-based clause store for subsumption checks

``Dissolve.solve()`` schedules its workers with a thread pool; ``engine.py``
and ``scheduler.py`` are used by ``dissolve_demo.py`` and are not instantiated
by the main loop.

Key classes
-----------

* ``Dissolve`` -- top-level solver; call ``solve(cnf)`` to run
* ``DissolveConfig`` -- configuration dataclass controlling solver behaviour:

  * ``k_split_vars`` -- number of variables used in dilemma splits
  * ``budget_strategy`` -- ``"constant"`` or ``"luby"`` sequence
  * ``max_shared_per_round`` -- cap on learned clauses shared per round

  The dataclass also defines ``distribution_strategy``,
  ``store_small_clauses_threshold``, ``per_query_conflict_budget``,
  ``important_bucket_size``, and ``kprime_for_decision_votes``. These fields are
  currently not read by ``Dissolve.solve()``: the solver always uses
  dilemma-rule splits and the UBTree store's built-in tier thresholds.

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
   from aria.utils.types import SolverResult

   cfg = DissolveConfig(
       k_split_vars=5,
       budget_strategy="luby",
   )
   solver = Dissolve(cfg)
   result = solver.solve(cnf_clauses)

   if result.result is SolverResult.SAT:
       print("Model:", result.model)
