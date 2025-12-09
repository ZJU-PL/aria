Constrained Horn Clauses (CHC) Tools
====================================

================
Introduction
================

Constrained Horn Clauses (CHCs) are a fragment of first-order logic that has become increasingly important in program verification and automated reasoning. The ``aria/quant/chctools`` module provides a comprehensive toolkit for working with CHCs, including parsing, solving, model validation, and pretty-printing.

A Constrained Horn Clause has the form:

.. math::

   \forall \vec{x}. (\phi(\vec{x}) \land P_1(\vec{t_1}) \land \ldots \land P_k(\vec{t_k})) \Rightarrow P(\vec{t})

where:
- :math:`\phi(\vec{x})` is a constraint over variables :math:`\vec{x}`
- :math:`P_1, \ldots, P_k, P` are uninterpreted predicates
- :math:`\vec{t_1}, \ldots, \vec{t_k}, \vec{t}` are terms

CHCs are particularly useful for program verification because they can naturally encode:
- **Invariants**: Properties that hold at program points
- **Pre/Post-conditions**: Function specifications
- **Transition relations**: Program state changes
- **Safety properties**: "Bad" states are unreachable

=======================
Core Components
=======================

The CHC tools module consists of several key components:

1. **Parser** (``parser.py``): Parses CHC problems from SMT-LIB format
2. **Horn Database** (``horndb.py``): Manages collections of Horn clauses and queries
3. **Solver** (``chcsolve.py``): Solves CHC problems using various backends
4. **Model Validator** (``chcmodel.py``): Validates models against CHC constraints
5. **Pretty Printer** (``chcpp.py``): Formats and outputs CHC problems
6. **Core Utilities** (``core.py``): Common command-line interface utilities

=======================
CHC Parser
=======================

The ``ChcRulesSmtLibParser`` class extends the standard SMT-LIB parser to handle CHC-specific constructs:

**Supported Commands:**
- ``declare-rel``: Declares uninterpreted predicates
- ``declare-var``: Declares variables
- ``rule``: Defines Horn clauses
- ``query``: Specifies queries to be solved

**Example CHC Problem:**

.. code-block:: smt

   (set-logic HORN)
   (declare-rel P (Int))
   (declare-rel Q (Int Int))
   (declare-var x Int)
   (declare-var y Int)
   
   ; Fact: P(0) is true
   (rule (P 0))
   
   ; Rule: P(x) ∧ x ≥ 0 → P(x+1)
   (rule (=> (and (P x) (>= x 0)) (P (+ x 1))))
   
   ; Rule: P(x) ∧ P(y) → Q(x, y)
   (rule (=> (and (P x) (P y)) (Q x y)))
   
   ; Query: Is Q(5, 10) reachable?
   (query (Q 5 10))

**Example Usage:**

.. code-block:: python

   from aria.quant.chctools.parser import ChcRulesSmtLibParser
   
   parser = ChcRulesSmtLibParser()
   with open('problem.smt2', 'r') as f:
       rules, queries = parser.get_chc(f)

=======================
Horn Database
=======================

The ``HornClauseDb`` class manages collections of Horn clauses and provides various operations:

**Key Classes:**

- ``HornRule``: Represents individual Horn clauses
- ``HornRelation``: Represents uninterpreted predicates
- ``HornClauseDb``: Container for rules and queries

**Rule Properties:**
- **Fact**: A rule with no uninterpreted predicates in the body
- **Linear**: A rule with at most one uninterpreted predicate in the body
- **Query**: A rule with ``false`` as the head

**Example Usage:**

.. code-block:: python

   from aria.quant.chctools.horndb import load_horn_db_from_file
   
   # Load CHC problem from file
   db = load_horn_db_from_file('problem.smt2')
   
   # Access rules and queries
   for rule in db.get_rules():
       print(f"Rule: {rule}")
       print(f"Is fact: {rule.is_fact()}")
       print(f"Is linear: {rule.is_linear()}")
   
   for query in db.get_queries():
       print(f"Query: {query}")

=======================
CHC Solver
=======================

The ``ChcSolveCmd`` class provides a command-line interface for solving CHC problems with various options:

**Solver Backends:**
- ``fp``: Uses Z3's Fixedpoint engine directly
- ``cli``: Generates command-line calls to Z3
- ``smt``: Converts to SMT format

**Key Options:**
- ``--pp``: Enable preprocessing (slicing, inlining)
- ``--st``: Print solving statistics
- ``--fresh``: Use fresh Z3 context
- ``--spctr FILE``: Generate Spacer trace file
- ``-y FILE``: Load configuration from YAML file

**Example Usage:**

.. code-block:: bash

   # Solve CHC problem with preprocessing
   python -m aria.quant.chctools.chcsolve --pp --st problem.smt2
   
   # Use custom Z3 options
   python -m aria.quant.chctools.chcsolve problem.smt2 spacer.mbqi=false

**Programmatic Usage:**

.. code-block:: python

   from aria.quant.chctools.chcsolve import chc_solve_with_fp
   from aria.quant.chctools.horndb import load_horn_db_from_file
   
   db = load_horn_db_from_file('problem.smt2')
   opts = {'spacer.mbqi': False}
   result = chc_solve_with_fp(db, args, opts)
   print(result)  # sat, unsat, or unknown

=======================
Model Validation
=======================

The ``ModelValidator`` class verifies that a given model satisfies all Horn clauses:

**Features:**
- Validates models against all rules and queries
- Provides detailed error reporting for invalid models
- Supports models in SMT-LIB ``define-fun`` format

**Example Usage:**

.. code-block:: python

   from aria.quant.chctools.chcmodel import ModelValidator, load_model_from_file
   from aria.quant.chctools.horndb import load_horn_db_from_file
   
   # Load problem and model
   db = load_horn_db_from_file('problem.smt2')
   model = load_model_from_file('model.smt2')
   
   # Validate model
   validator = ModelValidator(db, model)
   is_valid = validator.validate()
   print(f"Model is valid: {is_valid}")

**Command-line Usage:**

.. code-block:: bash

   python -m aria.quant.chctools.chcmodel -m model.smt2 problem.smt2

=======================
Pretty Printer
=======================

The ``ChcPpCmd`` class formats CHC problems for output:

**Output Formats:**
- ``rules``: Standard CHC format with ``rule`` and ``query`` commands
- ``chc``: SMT-LIB format with assertions

**Example Usage:**

.. code-block:: bash

   # Pretty-print as CHC rules
   python -m aria.quant.chctools.chcpp --format rules -o output.smt2 input.smt2
   
   # Convert to SMT format
   python -m aria.quant.chctools.chcpp --format chc -o output.smt2 input.smt2

**Programmatic Usage:**

.. code-block:: python

   from aria.quant.chctools.chcpp import pp_chc
   from aria.quant.chctools.horndb import load_horn_db_from_file
   
   db = load_horn_db_from_file('input.smt2')
   with open('output.smt2', 'w') as f:
       pp_chc(db, f, fmt='rules')

=======================
Advanced Features
=======================

**Query Splitting:**
Complex queries can be automatically split into simpler forms:

.. code-block:: python

   rule = HornRule(formula)
   if rule.is_query() and not rule.is_simple_query():
       simple_query, new_rule = rule.split_query()

**Solver Context Management:**
The ``pushed_solver`` utility provides safe solver state management:

.. code-block:: python

   from aria.quant.chctools.solver_utils import pushed_solver
   import z3
   
   solver = z3.Solver()
   solver.add(constraint1)
   
   with pushed_solver(solver) as s:
       s.add(constraint2)  # Temporary constraint
       result = s.check()
   # constraint2 is automatically removed

**Configuration Management:**
YAML configuration files can specify solver options:

.. code-block:: yaml

   spacer_opts:
     spacer.mbqi: false
     spacer.ground_pobs: false
     spacer.reach_dnf: true

=======================
Integration Examples
=======================

**Basic CHC Solving Workflow:**

.. code-block:: python

   from aria.quant.chctools.horndb import load_horn_db_from_file
   from aria.quant.chctools.chcsolve import chc_solve_with_fp
   import z3
   
   # Load CHC problem
   db = load_horn_db_from_file('problem.smt2')
   
   # Configure solver options
   opts = {
       'spacer.mbqi': False,
       'spacer.ground_pobs': False
   }
   
   # Solve the problem
   result = chc_solve_with_fp(db, args, opts)
   
   if result == z3.sat:
       print("Problem is satisfiable")
   elif result == z3.unsat:
       print("Problem is unsatisfiable")
   else:
       print("Result is unknown")

**Model Extraction and Validation:**

.. code-block:: python

   import z3
   from aria.quant.chctools.horndb import load_horn_db_from_file
   
   # Load and solve CHC problem
   db = load_horn_db_from_file('problem.smt2')
   fp = z3.Fixedpoint(ctx=db.get_ctx())
   db.mk_fixedpoint(fp=fp)
   
   # Solve and extract model if satisfiable
   for query in db.get_queries():
       result = fp.query(query.mk_query())
       if result == z3.sat:
           # Extract and save model
           model = fp.get_answer()
           print(f"Model: {model}")

=======================
Command-Line Tools
=======================

The CHC tools can be used directly from the command line:

**Solving CHC Problems:**

.. code-block:: bash

   # Basic solving
   python -m aria.quant.chctools.chcsolve problem.smt2
   
   # With preprocessing and statistics
   python -m aria.quant.chctools.chcsolve --pp --st problem.smt2
   
   # Custom solver options
   python -m aria.quant.chctools.chcsolve problem.smt2 spacer.mbqi=false spacer.reach_dnf=true

**Model Validation:**

.. code-block:: bash

   python -m aria.quant.chctools.chcmodel -m model.smt2 problem.smt2

**Pretty Printing:**

.. code-block:: bash

   python -m aria.quant.chctools.chcpp -o formatted.smt2 problem.smt2

=======================
Related Work and References
=======================

Constrained Horn Clauses have been extensively studied in the context of program verification:

- **Bjørner, N., Gurfinkel, A., McMillan, K., & Rybalchenko, A.** (2015). Horn clause solvers for program verification. In *Fields of Logic and Computation II* (pp. 24-51).
- **De Angelis, E., Fioravanti, F., Pettorossi, A., & Proietti, M.** (2014). Program verification via iterated specialization. *Science of Computer Programming*, 95, 149-175.
- **Grebenshchikov, S., Lopes, N. P., Popeea, C., & Rybalchenko, A.** (2012). Synthesizing software verifiers from proof rules. In *ACM SIGPLAN Conference on Programming Language Design and Implementation* (pp. 405-416).
- **Hoder, K., & Bjørner, N.** (2012). Generalized property directed reachability. In *International Conference on Theory and Applications of Satisfiability Testing* (pp. 157-171).

The Z3 theorem prover's Spacer engine is a state-of-the-art CHC solver that implements many advanced techniques for solving Horn clauses efficiently.
