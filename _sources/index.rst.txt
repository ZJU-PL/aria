Welcome to aria's Documentation!
=================================


=============
Introduction
=============

Aria is a comprehensive toolkit for automated reasoning and constraint solving. It provides implementations of various algorithms and tools for:

* **Abductive Inference** (``aria/abduction``) - Generate explanations for observations
* **AllSMT** (``aria/allsmt``) - Enumerate all satisfying models
* **Backbone Computation** (``aria/backbone``) - Extract forced assignments
* **UNSAT Core Extraction** (``aria/unsat_core``) - Identify minimal unsatisfiable subsets
* **Quantifier Reasoning** (``aria/quant``) - Handle exists-forall and quantified formulas
* **Quantifier Elimination** (``aria/quant/qe``) - Eliminate quantifiers from formulas
* **Solution Sampling** (``aria/sampling``) - Generate diverse solutions
* **Model Counting** (``aria/counting``) - Count satisfying assignments
* **Optimization Modulo Theory** (``aria/optimization``) - Solve optimization problems
* **Interpolant Generation** (``aria/interpolant``) - Generate Craig interpolants
* **Symbolic Abstraction** (``aria/symabs``) - Abstract state spaces
* **Predicate Abstraction** (``aria/symabs/predicate_abstraction``) - Abstract with predicates
* **Monadic Abstraction** (``aria/monabs``) - Monadic predicate abstraction
* **Knowledge Compilation** (``aria/bool/knowledge_compiler``) - Compile to tractable forms
* **MaxSAT Solving** (``aria/bool/maxsat``) - Solve maximum satisfiability problems
* **QBF Solving** - Quantified Boolean formula solving
* **Finite Field Solving** (``aria/smt/ff``) - SMT for Galois field constraints
* **Interactive Theorem Proving** (``aria/itp``) - Proof assistant framework
* **LLM Integration** (``aria/llm``) - Language model enhanced reasoning
* **Automata Operations** (``aria/automata``) - Finite automata algorithms
* **Program Synthesis** (``aria/synthesis``) - Synthesize programs from specifications
* **Context-Free Language Reachability** (``aria/cfl``) - CFL solving algorithms
* **Unification** (``aria/unification``) - Term unification algorithms
* **Translator** (``aria/translator``) - Translate between different formats

We welcome any feedback, issues, or suggestions for improvement. Please feel free to open an issue in our repository.

==========================
Installing and Using Aria
==========================

Install aria from source
---------------------------------------

::

  git clone https://github.com/ZJU-PL/aria
  virtualenv --python=python3 venv
  source venv/bin/activate
  cd aria
  bash setup_local_env.sh
  pip install -e .

The setup script will:
- Create a Python virtual environment if it doesn't exist
- Activate the virtual environment and install dependencies from requirements.txt
- Download required solver binaries (CVC5, MathSAT, z3)
- Run unit tests if available

Quick Start
-----------

::

  from aria import *

  # Example: Check satisfiability
  formula = Bool(True)  # Simple tautology
  result = smt_solve(formula)
  print(f"Formula is {'satisfiable' if result else 'unsatisfiable'}")

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   topics
   applications
   abduction
   allsmt
   automata
   backbone
   cfl
   chctools
   counting
   ff
   itp
   llm
   monabs
   optimization
   pcdclt
   polyhorn
   prob
   quantifiers
   sampling
   smt
   srk
   symbolic_abstraction
   symautomata
   synthesis
   unification
   unsat_core
