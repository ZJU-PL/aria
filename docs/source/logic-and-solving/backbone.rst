Backbone Computation
====================

Introduction to Backbone Computation
------------------------------------

The backbone of a Boolean formula is the set of literals that take the same value (True or False) in all satisfying assignments.
These literals represent the "fixed" or "necessary" parts of any solution to the formula.

Backbone computation has important applications in:

- **Constraint solving**: Identifying variables that must take specific values
- **SAT solving**: Simplifying formulas by fixing backbone literals
- **Verification**: Finding invariants in system models
- **Optimization**: Reducing search space by fixing necessary assignments

Backbone Computation in Aria
----------------------------

Aria provides backbone computation across Boolean (SAT) and SMT levels:

1. **Boolean SAT Backbone** (``aria.bool.backbone`` / ``aria.backbone.sat_backbone``):
   Computes the backbone of a CNF formula with multiple algorithms (``ITERATIVE``, ``CHUNKING``, ``BACKBONE_REFINEMENT``).

2. **SMT Backbone** (``aria.backbone.smt_backbone_literals`` and ``aria.backbone.smt_backbone_clauses``):
   Computes implied literals and clauses for SMT formulas. The literal-level
   dispatcher currently implements ``sequence_checking`` and
   ``model_enumeration``; the ``unsat_core_enumeration`` and monadic
   predicate-abstraction helpers are placeholders that raise
   ``NotImplementedError``.

Usage Example
-------------

.. code-block:: python

   from pysat.formula import CNF
   from aria.backbone import compute_backbone, BackboneAlgorithm

   # Boolean CNF backbone
   cnf = CNF(from_clauses=[[1, 2], [-1, 3], [-2, 3]])
   backbone, calls = compute_backbone(cnf, algorithm=BackboneAlgorithm.ITERATIVE)
   print("Backbone literals:", backbone)

   # SMT literal backbone
   import z3
   from aria.backbone.smt_backbone_literals import get_backbone_literals

   x, y, z = z3.Bools('x y z')
   formula = z3.And(z3.Or(x, y), z3.Or(z, z3.Not(y)), z3.Or(z3.Not(x), z3.Not(z)))
   literals = [x, y, z]
   smt_backbone = get_backbone_literals(formula, literals, alg="sequence_checking")
   print("SMT backbone literals:", smt_backbone)

Algorithm Details
-----------------

The backbone computation algorithms in Aria use several optimizations:

- **Incremental SAT solving**: Reusing solver state between iterations
- **Assumption-based solving**: Efficiently testing candidate backbone literals
- **Clause learning**: Leveraging learned clauses to speed up subsequent iterations
- **Early termination**: Detecting when no more backbone literals can be found

References
----------

- I. Lynce and J. Marques-Silva. "Efficient haplotype inference with Boolean satisfiability." National Conference on Artificial Intelligence, 2006.
- J. Marques-Silva, M. Janota, and I. Lynce. "On computing backbones of propositional theories." ECAI, 2010.
- N. Narodytska and T. Walsh. "The importance of being structured: On the computation of backdoors for MaxSAT." AAAI, 2014.
