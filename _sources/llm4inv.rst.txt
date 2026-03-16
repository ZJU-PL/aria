LLM4Inv: LLM-Guided Invariant Synthesis
==========================================

Overview
--------

LLM4Inv is a verification engine that uses Large Language Models (LLMs) to synthesize program invariants through a guess-and-check approach. The engine implements a CEGIS (Counter-Example Guided Inductive Synthesis) loop where:

1. The LLM proposes concrete, hole-free invariant candidates
2. An SMT solver verifies each candidate for validity
3. Failed candidates and counterexamples are used to refine subsequent proposals
4. The process iterates until a valid invariant is found or a timeout is reached

This approach is particularly effective for bitvector programs where the LLM can leverage its understanding of program semantics to generate meaningful invariants.

Architecture
------------

The LLM4Inv engine consists of three main components:

- **LLM4InvProver**: The main interface that wraps the CEGIS loop
- **LLMInvariantCEGIS**: Implements the iterative synthesis loop
- **LLMInterface**: Provides abstraction for LLM communication (supports both local and remote models)

Usage
-----

Basic usage:

.. code-block:: bash

   efmc --engine llm4inv --lang chc --file program.smt2

With custom LLM model:

.. code-block:: bash

   efmc --engine llm4inv --lang chc --file program.smt2 \
        --llm4inv-provider remote \
        --llm4inv-remote-model deepseek-v3

Using local LLM:

.. code-block:: bash

   efmc --engine llm4inv --lang chc --file program.smt2 \
        --llm4inv-provider local \
        --llm4inv-local-provider lm-studio \
        --llm4inv-local-model qwen/qwen3-coder-30b

Command-Line Options
--------------------

The LLM4Inv engine supports the following command-line options:

- ``--llm4inv-max-candidates-per-iter``: Maximum number of candidates per iteration (default: 5)
- ``--llm4inv-provider``: LLM provider - ``local`` or ``remote`` (default: local)
- ``--llm4inv-local-provider``: Local provider - ``lm-studio``, ``vllm``, or ``sglang`` (default: lm-studio)
- ``--llm4inv-local-model``: Local model name (default: qwen/qwen3-coder-30b)
- ``--llm4inv-remote-model``: Remote model name - ``deepseek-v3``, ``glm-4-flash``, etc. (default: deepseek-v3)
- ``--llm4inv-temperature``: Temperature for LLM generation (default: 0.1)
- ``--llm4inv-max-output-length``: Maximum output length (default: 4096)
- ``--llm4inv-measure-cost``: Measure token costs (default: False)
- ``--llm4inv-max-iterations``: Maximum number of CEGIS iterations (default: 10)

How It Works
------------

1. **Program Analysis**: The engine extracts program information including variables, bit widths, and transition relations.

2. **Prompt Generation**: A prompt is generated that includes:
   - Program description
   - Variable information
   - Previously failed candidates (for refinement)
   - Counterexamples from verification

3. **Candidate Generation**: The LLM generates multiple invariant candidates in SMT-LIB2 format.

4. **Verification**: Each candidate is verified using SMT solving:
   - Init ⇒ Inv (initial states satisfy invariant)
   - Inv ∧ Trans ⇒ Inv' (invariant is inductive)
   - Inv ⇒ Post (invariant implies safety property)

5. **Refinement**: If verification fails, counterexamples are added to the prompt for the next iteration.

6. **Termination**: The loop terminates when:
   - A valid invariant is found
   - Maximum iterations are reached
   - Timeout is exceeded

Supported Models
----------------

**Remote Models:**
- DeepSeek V3
- GLM-4-Flash
- Other OpenAI-compatible API models

**Local Models:**
- Qwen3-Coder-30B
- Other models supported by LM Studio, vLLM, or SGLang

Limitations
-----------

- Currently optimized for bitvector programs
- Requires LLM API access (for remote models) or local LLM server
- Performance depends on LLM quality and prompt engineering
- May require multiple iterations for complex invariants

Related Work
------------

- Counter-Example Guided Inductive Synthesis (CEGIS)
- LLM-guided program synthesis
- Neural-guided invariant generation

