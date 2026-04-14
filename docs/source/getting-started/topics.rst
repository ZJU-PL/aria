Research Topics and Thesis Projects
===================================

ARIA offers many research directions across automated reasoning, verification,
and solver engineering.

Core algorithm development
--------------------------

**Parallel CDCL(T) solving** (``aria/smt/pcdclt``)
  Develop parallel algorithms for clause learning with theory reasoning.

**Optimization modulo theory** (``aria/pyomt``)
  Extend OMT algorithms over bit-vectors, arithmetic, and mixed settings.

**Advanced model counting** (``aria/counting``)
  Improve counting algorithms for Boolean, arithmetic, and QF_BV formulas.

**Symbolic abstraction** (``aria/symabs``)
  Develop abstraction techniques for infinite-state systems and verification.

Theory-specific solving
-----------------------

**Finite-field SMT** (``aria/smt/ff``)
  Build decision procedures for Galois-field constraints.

**Floating-point arithmetic** (``aria/smt/fp``)
  Develop efficient IEEE-754 solving and optimization workflows.

AI-enhanced reasoning
---------------------

**Machine learning for solvers** (``aria/ml``)
  Learn solver selection, feature extraction, and tactic guidance.

**Automata learning** (``aria/automata``)
  Apply learning-based techniques to string solving and verification.

Advanced sampling and enumeration
---------------------------------

**Uniform sampling** (``aria/sampling``)
  Design diverse solution-sampling strategies over rich SMT theories.

**AllSMT algorithms** (``aria/allsmt``)
  Improve exhaustive model enumeration and projection workflows.

Quantifier handling
-------------------

**Quantifier elimination** (``aria/quant/qe``)
  Study QE procedures for arithmetic and related theories.

**E-matching optimization** (``aria/ml/llm/ematching``)
  Improve trigger selection and quantifier instantiation heuristics.

**CHC solving** (``aria/quant/chctools``)
  Scale constrained Horn clause solving for verification and synthesis.

Applications and tools
----------------------

**Interactive theorem proving** (``aria/itp``)
  Build proof tools across multiple theories.

**Program synthesis** (``aria/synthesis``)
  Explore SyGuS and related synthesis workflows.

**Abductive reasoning** (``aria/abduction``)
  Develop explanation and hypothesis-generation algorithms.

Getting started
---------------

Start with the package README files under ``aria/`` and the section pages in
this documentation tree, then drill down into the subsystem that matches your
interests.
