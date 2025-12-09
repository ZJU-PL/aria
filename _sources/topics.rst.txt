Research Topics and Thesis Projects
===================================

Aria offers numerous opportunities for research and thesis projects across multiple areas of automated reasoning.

=========
Core Algorithm Development
=========

**Parallel CDCL(T) Solving** (``aria/smt/pcdclt``)
  Develop parallel algorithms for conflict-driven clause learning with theory reasoning. Focus on work distribution, clause sharing, and portfolio solving.

**Optimization Modulo Theory** (``aria/optimization``)
  Extend SMT solving with optimization capabilities. Implement algorithms for OMT over bit-vectors, arithmetic, and mixed theories.

**Advanced Model Counting** (``aria/counting``)
  Improve counting algorithms for Boolean, arithmetic, and quantifier-free bit-vector formulas. Focus on scalability and approximation techniques.

**Symbolic Abstraction** (``aria/symabs``)
  Develop new abstraction techniques for infinite state systems. Implement counterexample-guided abstraction refinement (CEGAR).

=========
Theory-Specific Solving
=========

**Finite Field SMT** (``aria/smt/ff``)
  Build decision procedures for Galois field constraints. Applications in cryptography and coding theory.

**Floating-Point Arithmetic** (``aria/smt/fp``)
  Develop efficient solvers for IEEE 754 floating-point constraints with proper handling of rounding modes and special values.

**String Constraint Solving**
  Extend string theory support with automata-based techniques. Implement length constraints and regular language operations.

=========
AI-Enhanced Reasoning
=========

**LLM-Driven Constraint Solving** (``aria/llm``)
  Integrate large language models to guide solver heuristics, strategy selection, and formula preprocessing.

**Machine Learning for Solvers** (``aria/ml``)
  Extract features for learned solver selection, clause learning prediction, and variable ordering heuristics.

**Automata Learning** (``aria/automata``)
  Apply active learning to infer automata from examples for string constraint solving and program verification.

=========
Advanced Sampling & Enumeration
=========

**Uniform Sampling** (``aria/sampling``)
  Develop algorithms for uniform solution sampling over complex constraint domains. Applications in probabilistic verification.

**AllSMT Algorithms** (``aria/allsmt``)
  Enumerate all solutions efficiently. Focus on diversity metrics and incremental solving techniques.

**Solution Space Analysis**
  Implement tools for analyzing solution spaces, including backbone computation and minimal unsatisfiable core extraction.

=========
Quantifier Handling
=========

**Quantifier Elimination** (``aria/quant/qe``)
  Develop QE procedures for mixed theories combining arithmetic, bit-vectors, and arrays.

**E-Matching Optimization** (``aria/quant/ematching``)
  Improve quantifier instantiation through better pattern matching and trigger selection.

**CHC Solving** (``aria/quant/chctools``)
  Scale algorithms for constrained Horn clause solving. Applications in program verification and synthesis.

=========
Applications & Tools
=========

**Interactive Theorem Proving** (``aria/itp``)
  Build proof assistant tools with support for multiple theories and automated proof search.

**Program Synthesis** (``aria/synthesis``)
  Implement syntax-guided synthesis techniques for bit-vectors, arithmetic, and string domains.

**Abductive Reasoning** (``aria/abduction``)
  Develop algorithms for generating explanations and hypotheses from constraint observations.

=========
Getting Started
=========

Each module includes examples and documentation. Start with ``aria/allsmt`` for basic usage patterns, then explore specialized areas based on your interests.
