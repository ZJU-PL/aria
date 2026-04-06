Exists-Forall SMT (EFSMT)
========================

The ``aria.quant`` package contains several implementations for
exists-forall reasoning, together with shared parsing and solver glue.
Rather than exposing one single EFSMT engine, the repository organizes this
area into theory-specific stacks and a small shared front end.

Overview
--------

EFSMT problems have the form ``exists x . forall y . phi(x, y)``. In ARIA,
the corresponding code lives mainly in ``aria/quant`` and spans pure Boolean,
bit-vector, and linear-arithmetic fragments.

The key shared files are:

- ``aria/quant/efsmt_parser.py``: parses EFSMT-style SMT-LIB inputs and
  extracts existential variables, universal variables, and the matrix.
- ``aria/quant/efsmt_solver.py``: generic solver wrapper and EFSMT-facing
  orchestration utilities.
- ``aria/quant/efsmt_utils.py``: helper routines for instantiation and solver
  interaction.

Theory-Specific Solver Families
-------------------------------

``aria.quant`` keeps separate exists-forall stacks for different theories.
This separation is intentional: the directory mixes reusable front-end code,
theory-specific solvers, and research prototypes with different maturity
levels.

Boolean EFSMT
~~~~~~~~~~~~~

``aria/quant/efbool`` implements exists-forall solving over pure Boolean
formulas. The package includes separate existential and universal solving
components plus sequential and parallel utilities for counterexample-guided
refinement.

Representative files include:

- ``efbool_seq.py``
- ``efbool_exists_solver.py``
- ``efbool_forall_solver.py``
- ``efbool_parallel_utils.py``

Bit-Vector EFSMT
~~~~~~~~~~~~~~~~

``aria/quant/efbv`` is the main bit-vector exists-forall stack. It combines
several solving strategies instead of one fixed backend:

- ``efbv_seq/`` contains sequential engines, including direct SMT solving,
  QBF reduction, SAT reduction, and CEGIS-style workflows.
- ``efbv_parallel/`` contains parallel variants with candidate generation,
  counterexample checking, and sampling-oriented refinement.

This makes ``efbv`` a good starting point for understanding how ARIA explores
multiple implementation strategies for the same quantified fragment.

Linear-Arithmetic EFSMT
~~~~~~~~~~~~~~~~~~~~~~~

``aria/quant/eflira`` targets exists-forall problems over linear arithmetic.
The implementations use counterexample-guided refinement with paired solver
instances, together with sequential and parallel workflows.

Representative files include:

- ``eflira_seq.py``
- ``eflira_parallel.py``

Related Quantified Reasoning Code
---------------------------------

Some nearby packages are closely related to EFSMT, even when they are not the
main shared pipeline:

- ``aria/quant/ufbv`` explores quantified bit-vector solving via parallel
  under- and over-approximation.
- ``aria/quant/qe`` contains quantifier-elimination experiments and adapters.
- ``aria/quant/chctools`` and ``aria/quant/polyhorn`` cover adjacent quantified
  reasoning workflows built around Horn clauses and solver experimentation.

Practical Notes
---------------

- ``aria.quant`` is intentionally heterogeneous; similar ideas may appear in
  more than one implementation.
- Some solvers depend on optional external binaries or solver configurations.
- The best entry point depends on the target theory: ``efbool`` for pure
  Boolean formulas, ``efbv`` for bit-vectors, and ``eflira`` for linear
  arithmetic.

Further Reading
---------------

- ``aria/quant/README.md`` for a package-level map
- ``aria/quant/AGENTS.md`` for local development guidance and testing notes
