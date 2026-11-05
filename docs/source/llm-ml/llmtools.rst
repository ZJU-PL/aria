.. _llmtools:

LLM Tools
=========

``aria.ml.llmtools`` is ARIA's provider and routing layer for LLM inference across
online and local backends. It is also used by verification workflows such as the
LLM4Inv engine, but it is no longer nested under ``aria.efmc``.

Package layout
--------------

The current package structure is:

.. code-block:: text

   aria/ml/llmtools/
   ├── __init__.py
   ├── client.py
   ├── local_client.py
   ├── routing.py
   ├── tooling.py
   ├── core/
   └── providers/

Key modules:

* ``client.py``: routed ``LLM`` client
* ``local_client.py``: local-provider client helpers
* ``routing.py``: provider/model resolution
* ``tooling.py``: abstract base classes for LLM-backed tools
* ``core/``: shared types, logging, retry logic, and client flow
* ``providers/``: online and local provider adapters

Public API
----------

The package exports:

.. code-block:: python

   from aria.ml.llmtools import LLM, LLMLocal, Logger, resolve_provider

Example
-------

.. code-block:: python

   from aria.ml.llmtools import LLM, Logger

   logger = Logger("logs/llm.log")  # path to the log file
   llm = LLM(
       model_name="gpt-4o-mini",
       logger=logger,
       temperature=0.1,
   )

The concrete provider is selected by ``resolve_provider()`` based on the model
name and optional provider hint.

LLM-backed tool base classes
----------------------------

``aria.ml.llmtools.tooling`` defines reusable abstractions for higher-level tools:

.. code-block:: python

   from aria.ml.llmtools.tooling import LLMTool, LLMToolInput, LLMToolOutput

These classes provide:

* a typed input/output boundary for tool wrappers
* retry-aware invocation through the routed ``LLM`` client
* prompt construction and response parsing hooks
* simple caching keyed by tool input

Relationship to EFMC
--------------------

Verification engines may consume ``aria.ml.llmtools``, but the LLM infrastructure
is shared repo-wide and should be documented at the top-level package location.

API reference
-------------

.. automodule:: aria.ml.llmtools
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.ml.llmtools.client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.ml.llmtools.local_client
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.ml.llmtools.routing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aria.ml.llmtools.tooling
   :members:
   :undoc-members:
   :show-inheritance:
