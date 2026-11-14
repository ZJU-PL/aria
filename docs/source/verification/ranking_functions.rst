Ranking Functions
=================

Ranking functions are used to prove termination of bit-vector loops: a
function ``R(x)`` that is bounded below and strictly decreases along every
loop iteration witnesses termination. The dual notion, *recurrence sets*,
witnesses non-termination.

Implementation
--------------

The concrete implementation lives in
``aria.efmc.engines.ef.termination``. It provides template-based termination
and non-termination analysis for bit-vector transition systems:

* ``TerminationProver`` -- generates ranking-function (or recurrence-set)
  verification conditions from a ``TransitionSystem``
* ``RankingSolver`` / ``RecurrenceSolver`` -- solve the generated templates
* ``RankingFunctionValidator`` / ``RecurrenceSetValidator`` -- independently
  check a candidate witness

High-level API
--------------

``aria.efmc.engines.ef.termination`` exports three convenience functions:

* ``prove_termination_with_ranking_functions(sts, template_names=None,
  timeout=None, **kwargs)`` -- returns ``(proven, ranking_function, template)``.
  By default it tries the templates ``bv_linear_ranking``,
  ``bv_lexicographic_ranking``, and ``bv_conditional_ranking`` in order.
* ``prove_non_termination_with_recurrence_sets(sts, template_names=None,
  timeout=None, **kwargs)`` -- returns ``(proven, recurrence_set, template)``;
  default templates are ``bv_linear_recurrence``,
  ``bv_interval_recurrence``, and ``bv_disjunctive_recurrence``.
* ``analyze_termination(sts, ranking_templates=None, recurrence_templates=None,
  timeout=None, **kwargs)`` -- tries the ranking templates first and only falls
  back to recurrence sets if termination could not be proven. It returns a
  dictionary with ``termination_proven``, ``non_termination_proven``,
  ``ranking_function``, ``recurrence_set``, the template names used, and an
  ``errors`` list.

Example
-------

.. code-block:: python

   import z3
   from aria.efmc.sts import TransitionSystem
   from aria.efmc.engines.ef.termination import analyze_termination

   x = z3.BitVec("x", 8)
   x_p = z3.BitVec("x_p", 8)
   sts = TransitionSystem(
       variables=[x],
       prime_variables=[x_p],
       init=(x == 0),
       trans=z3.And(z3.ULT(x, z3.BitVecVal(100, 8)), x_p == x + 1),
       post=(z3.ULT(x, z3.BitVecVal(100, 8))),
   )

   results = analyze_termination(sts)
   if results["termination_proven"]:
       print("Ranking function:", results["ranking_function"])
       print("Template:", results["ranking_template_used"])

Notes
-----

* The templates target bit-vector programs; the analysis assumes a
  ``TransitionSystem`` with ``init``, ``trans``, and ``post`` set.
* Failure to prove either direction leaves both flags ``False``; it is not a
  proof of the opposite property.
* Earlier versions of this page showed examples from unrelated modules
  (``aria.efmc.engines.ef`` ranking templates and the main EFMC CLI); those
  examples do not correspond to the termination package and have been removed.
