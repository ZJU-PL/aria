.. _llmtools:

LLM Tools
=========

The ``llmtools`` module provides utilities for integrating Large Language Models (LLMs) into the verification pipeline. This is primarily used by the LLM4Inv engine for generating candidate invariants.

Architecture
------------

The module follows an abstract base class pattern:

.. code-block::

   efmc/llmtools/
   ├── llm_tool.py      # Abstract base classes
   ├── llm_utils.py     # LLM wrapper and utilities
   ├── llm_local.py     # Local model support
   ├── llm_opencode.py  # OpenCode API integration
   ├── logger.py        # Logging utilities
   └── test_llm.py      # Test utilities

LLMTool Base Classes
--------------------

LLMToolInput
~~~~~~~~~~~~

Abstract base class for LLM tool inputs.

.. code-block:: python

   from efmc.llmtools.llm_tool import LLMToolInput

   class MyToolInput(LLMToolInput):
       def __init__(self, program: str, property: str):
           self.program = program
           self.property = property

       def __hash__(self):
           return hash((self.program, self.property))

LLMToolOutput
~~~~~~~~~~~~~

Abstract base class for LLM tool outputs.

.. code-block:: python

   from efmc.llmtools.llm_tool import LLMToolOutput

   class MyToolOutput(LLMToolOutput):
       def __init__(self, candidates: List[str]):
           self.candidates = candidates

LLMTool
~~~~~~~

Abstract base class for LLM-based tools.

.. code-block:: python

   from efmc.llmtools.llm_tool import LLMTool

   class MyLLMTool(LLMTool):
       def _get_prompt(self, tool_input: MyToolInput) -> str:
           # Generate prompt from input
           return f"Generate invariants for:\n{tool_input.program}"

       def _parse_response(self, response: str, tool_input: MyToolInput) -> MyToolOutput:
           # Parse LLM response
           return MyToolOutput(candidates=response.split("\n"))

LLM Utilities
-------------

The ``llm_utils`` module provides the core LLM interface:

.. code-block:: python

   from efmc.llmtools.llm_utils import LLM

   # Initialize LLM
   llm = LLM(
       model_name="qwen/qwen3-coder-30b",
       logger=my_logger,
       temperature=0.1
   )

   # Query the model
   response, input_tokens, output_tokens = llm.infer(
       prompt="Generate an invariant for x >= 0",
       verbose=True
   )

LLM Providers
-------------

Local Models
~~~~~~~~~~~~

Support for locally hosted models via various backends:

.. code-block:: python

   from efmc.llmtools.llm_local import LocalLLM

   # LM Studio
   local_llm = LocalLLM(
       provider="lm-studio",
       model="qwen/qwen3-coder-30b"
   )

   # vLLM
   local_llm = LocalLLM(
       provider="vllm",
       model="meta-llama/Llama-2-7b-chat-hf"
   )

   # SGLang
   local_llm = LocalLLM(
       provider="sglang",
       model="deepseek-ai/DeepSeek-V2"
   )

Remote Models
~~~~~~~~~~~~~

Support for remote API providers:

.. code-block:: python

   from efmc.llmtools.llm_opencode import OpenCodeLLM

   remote_llm = OpenCodeLLM(
       model="deepseek-v3",
       api_key="your-api-key"
   )

Logging
-------

The ``logger`` module provides structured logging for LLM interactions:

.. code-block:: python

   from efmc.llmtools.logger import Logger

   logger = Logger(
       log_file="llm.log",
       console_log=True,
       level="INFO"
   )

   logger.print_console("Starting LLM inference...")
   logger.print_log("Prompt:", prompt)
   logger.print_log("Response:", response)

Usage in LLM4Inv
----------------

The LLM4Inv engine uses these utilities to generate candidate invariants:

.. code-block:: python

   from efmc.engines.llm4inv import LLM4InvProver

   prover = LLM4InvProver(
       transition_system=sts,
       max_candidates=5,
       provider="local",
       model="qwen/qwen3-coder-30b"
   )

   result = prover.solve()

Configuration
-------------

LLM behavior can be configured via command-line arguments:

.. code-block:: bash

   efmc --file program.smt2 --engine llm4inv \
       --llm4inv-provider local \
       --llm4inv-local-provider lm-studio \
       --llm4inv-local-model qwen/qwen3-coder-30b \
       --llm4inv-temperature 0.1 \
       --llm4inv-max-candidates-per-iter 5 \
       --llm4inv-max-iterations 10

API Reference
-------------

.. automodule:: efmc.llmtools.llm_tool
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.llmtools.llm_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.llmtools.llm_local
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.llmtools.llm_opencode
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: efmc.llmtools.logger
   :members:
   :undoc-members:
   :show-inheritance:
