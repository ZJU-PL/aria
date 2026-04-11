Abstract Interpretation
=======================

Abstract interpretation appears in ARIA both as general background and as
engine-specific code inside verification components.

Current codebase note
---------------------

Within EFMC, the ``aria.efmc.engines.absint`` directory currently contains
specialized affine-relation code rather than a single broad public
``AbstractInterpretationProver`` API.

Related package areas
---------------------

* ``aria.symabs.ai_symabs`` for abstract-interpretation-oriented symbolic abstraction
* ``aria.efmc.engines.absint`` for specialized verification-side components

Concepts
--------

Abstract interpretation computes sound over-approximations of reachable program
states by iterating abstract transformers to a fixpoint, often with widening or
other convergence accelerators.

References
----------

* Cousot and Cousot, abstract interpretation foundations
* the package-level docs and code under ``aria/symabs`` for current ARIA examples
