Interactive Theorem Proving (ITP) [Removed]
============================================

The Python ``aria.itp`` package (kernel, tactics, datatypes, theories, and the
Z3/CVC5/Vampire ``smt`` shim) was removed from the repository. It is no longer
importable: ``import aria.itp.smt`` and ``from aria.itp.theories import ...``
fail, and the related regression test is skipped with the message
"aria.itp module was removed from repository".

This page is kept so that existing links and references still resolve. It
documents the current state rather than the removed API.

What exists today
-----------------

* ``aria/itp/fstar-copilot/`` -- an F\* agent-skill bundle (prompts and
  benchmark tasks). It is data for coding agents, not an importable Python
  module, and it does not provide a proof kernel.
* ``aria.smt.fol`` -- the Miniprover first-order logic theorem prover; see
  :doc:`../logic-and-solving/fol`.
* ``aria.proof`` -- abductive reasoning, belief revision, and Craig
  interpolation helpers; see :doc:`abduction` and :doc:`interpolant`.
* ``aria.bool.interpolant`` -- Boolean interpolation from resolution proofs;
  see :doc:`../logic-and-solving/bool`.

Migrating old examples
----------------------

Old snippets that relied on ``aria.itp`` need to be rewritten against one of
the packages above, or dropped. In particular:

* ``prove``/``axiom``/``define``/``Proof`` have no replacement in ARIA; use a
  solver or the Miniprover directly.
* Formalized theories (``int``, ``set``, ``seq``, ``algebra``, ``real``) are
  no longer shipped.
* The bounded second-order decision procedure that lived in
  ``aria.itp.theories.logic.reverse_math_poc`` is gone.

The previous implementation is still available in the repository history
(commits before the ``rm itp`` change).
