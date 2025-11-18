Welcome to arlib's Documentation!
=================================


=============
Introduction
=============

Arlib is a comprehensive toolkit for automated reasoning and constraint solving. It provides implementations of various algorithms and tools for:

* **Abductive Inference** (``arlib/abduction``) - Generate explanations for observations
* **AllSMT** (``arlib/allsmt``) - Enumerate all satisfying models
* **Backbone Computation** (``arlib/backbone``) - Extract forced assignments
* **UNSAT Core Extraction** (``arlib/unsat_core``) - Identify minimal unsatisfiable subsets
* **Quantifier Reasoning** (``arlib/quant``) - Handle exists-forall and quantified formulas
* **Quantifier Elimination** (``arlib/quant/qe``) - Eliminate quantifiers from formulas
* **Solution Sampling** (``arlib/sampling``) - Generate diverse solutions
* **Model Counting** (``arlib/counting``) - Count satisfying assignments
* **Optimization Modulo Theory** (``arlib/optimization``) - Solve optimization problems
* **Interpolant Generation** (``arlib/interpolant``) - Generate Craig interpolants
* **Symbolic Abstraction** (``arlib/symabs``) - Abstract state spaces
* **Predicate Abstraction** (``arlib/symabs/predicate_abstraction``) - Abstract with predicates
* **Monadic Abstraction** (``arlib/monabs``) - Monadic predicate abstraction
* **Knowledge Compilation** (``arlib/bool/knowledge_compiler``) - Compile to tractable forms
* **MaxSAT Solving** (``arlib/bool/maxsat``) - Solve maximum satisfiability problems
* **QBF Solving** - Quantified Boolean formula solving
* **Finite Field Solving** (``arlib/smt/ff``) - SMT for Galois field constraints
* **Interactive Theorem Proving** (``arlib/itp``) - Proof assistant framework
* **LLM Integration** (``arlib/llm``) - Language model enhanced reasoning
* **Automata Operations** (``arlib/automata``) - Finite automata algorithms
* **Program Synthesis** (``arlib/synthesis``) - Synthesize programs from specifications
* **Context-Free Language Reachability** (``arlib/cfl``) - CFL solving algorithms
* **Unification** (``arlib/unification``) - Term unification algorithms

We welcome any feedback, issues, or suggestions for improvement. Please feel free to open an issue in our repository.

==========================
Installing and Using Arlib
==========================

Install arlib from source
---------------------------------------

::

  git clone https://github.com/ZJU-PL/arlib
  virtualenv --python=python3 venv
  source venv/bin/activate
  cd arlib
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

  from arlib import *

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
