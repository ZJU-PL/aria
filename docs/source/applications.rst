Applications
============

Aria supports various applications across testing, verification, synthesis, and optimization.

==========
Testing
==========

**Constrained Random Testing**
  Generate test cases satisfying logical constraints using ``aria/sampling`` and ``aria/allsmt``.

**Combinatorial Testing**
  Generate diverse test suites with ``aria/bool/features`` for covering parameter interactions.

==========
Verification
==========

**Predicate Abstraction**
  Abstract program states using ``aria/symabs/predicate_abstraction`` for verification.

**Symbolic Abstraction**
  Abstract infinite state spaces with ``aria/symabs`` for model checking.

**Interactive Theorem Proving**
  Formal verification with ``aria/itp`` framework supporting multiple theories.

==========
Synthesis
==========

**Program Synthesis**
  Synthesize programs from specifications using ``aria/synthesis`` (SyGuS, PBE).

**Syntax-Guided Synthesis**
  Generate programs matching given grammars with ``aria/synthesis/sygus_*``.

==========
Optimization
==========

**Optimization Modulo Theory**
  Solve optimization problems over logical theories using ``aria/optimization``.

**MaxSAT Solving**
  Solve maximum satisfiability problems with ``aria/bool/maxsat``.

==========
Learning & AI
==========

**Machine Learning Features**
  Extract features for ML-based solver selection with ``aria/ml``.
