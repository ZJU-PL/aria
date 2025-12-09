LLM Integration
===========================

Introduction
=====================

The LLM module (``aria/llm``) integrates large language models into automated reasoning workflows, enabling natural language interfaces, LLM-guided solving strategies, and reasoning about programs with unspecified components.

Key Features
-------------

* **Natural Language to SMT**: Convert natural language constraints to SMT formulas
* **SMTO (SMT + Oracles)**: Solve constraints with LLM-based oracle components
* **LLM-Guided Abduction**: Generate explanations using language models
* **Interpolant Generation**: Synthesize interpolants with LLM assistance
* **UNSAT Core Analysis**: Minimal unsatisfiable core extraction with explanations

Components
=====================

Natural Language Translation (``aria/llm/nl2smt.py``, ``aria/llm/smt2nl.py``)
--------------------------------------------------------------------------------

Bidirectional translation between natural language and SMT-LIB:

.. code-block:: python

   from aria.llm import nl_to_smt, smt_to_nl

   # Convert natural language to SMT
   formula = nl_to_smt("x is greater than 5 and less than 10")

   # Convert SMT to natural language description
   description = smt_to_nl(smt_formula)

SMTO: SMT with Oracles (``aria/llm/smto``)
--------------------------------------------

Satisfiability Modulo Theories and Oracles for reasoning about programs with unspecified components:

.. code-block:: python

   from aria.llm.smto import OraxSolver, OracleInfo
   import z3

   # Initialize solver
   solver = OraxSolver(provider="openai", model="gpt-4")

   # Register blackbox oracle with examples
   oracle = OracleInfo(
       name="check_password",
       input_types=[z3.StringSort()],
       output_type=z3.BoolSort(),
       description="Validate password strength",
       examples=[
           {"input": {"arg0": "weak"}, "output": "false"},
           {"input": {"arg0": "Str0ng!Pass"}, "output": "true"}
       ]
   )
   solver.register_oracle(oracle)

   # Solve constraints
   password = z3.String('password')
   check_pw = z3.Function('check_password', z3.StringSort(), z3.BoolSort())
   solver.add_constraint(check_pw(password) == True)
   model = solver.check()

**Whitebox Mode**: Enhanced analysis with source code or documentation:

.. code-block:: python

   from aria.llm.smto import WhiteboxOracleInfo, OracleAnalysisMode

   oracle = WhiteboxOracleInfo(
       name="oracle_func",
       analysis_mode=OracleAnalysisMode.SOURCE_CODE,
       source_code="def oracle_func(x): return x > 0 and x < 100"
   )

LLM-Guided Abduction (``aria/llm/abduct``)
--------------------------------------------

Generate explanatory hypotheses for observations:

.. code-block:: python

   from aria.llm.abduct import LLMAbductor

   abductor = LLMAbductor()
   explanation = abductor.abduce(
       background_theory=theory,
       observation=observation
   )

LLM Interpolant Generation (``aria/llm/interpolant``)
-------------------------------------------------------

Synthesize Craig interpolants using language models:

.. code-block:: python

   from aria.llm.interpolant import LLMInterpolant

   interpolant = LLMInterpolant().generate(formula_a, formula_b)

LLM Tools (``aria/llm/llmtool``)
----------------------------------

Utilities for LLM integration:

* Multi-provider support (OpenAI, Anthropic, local models)
* Prompt engineering utilities
* Response parsing and validation
* Caching and rate limiting

Applications
=====================

* Natural language specification to formal constraints
* Reasoning about third-party libraries without specifications
* LLM-guided solver heuristics and strategy selection
* Automated explanation generation for verification results
* Interactive constraint debugging with natural language

References
=====================
