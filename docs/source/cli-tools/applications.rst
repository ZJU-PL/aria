Applications
============

Aria supports various applications across testing, verification, synthesis, and optimization.

Testing
=======

**Constrained Random Testing**
  Generate test cases satisfying logical constraints using ``aria/sampling`` and ``aria/allsmt``.

**Concolic Testing**
  Perform native Python concolic testing with validated symbolic shadows using ``aria/concolic``.

Verification
============

**Predicate Abstraction**
  Abstract program states using ``aria/symabs/predicate_abstraction`` for verification.

**Symbolic Abstraction**
  Abstract infinite state spaces with ``aria/symabs`` for model checking.

**Transition System Verification**
  Verify safety properties of transition systems across CHC, SyGuS, Boogie, and C using ``aria/efmc``.

Synthesis
=========

**Program Synthesis**
  Synthesize programs from specifications using ``aria/synthesis`` (PBE, version space algebra).

**Syntax-Guided Synthesis**
  Generate programs matching given grammars with ``aria/synthesis/cvc5``.

Optimization
============

**Optimization Modulo Theory**
  Solve optimization problems over logical theories using ``aria/pyomt``.

**MaxSAT Solving**
  Solve maximum satisfiability problems with ``aria/bool/maxsat``.

Learning & AI
=============

**Machine Learning Features**
  Extract features for ML-based solver selection with ``aria/ml/machfea`` and ``aria/ml/smtgazer``.
