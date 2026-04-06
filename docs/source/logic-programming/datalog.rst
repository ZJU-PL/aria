Datalog
=======

The ``aria.datalog`` package provides a vendored ``pyDatalog`` runtime under
the ARIA namespace. It is a trimmed, runtime-oriented copy intended to keep
logic-programming support available inside the repository without depending on
an external top-level ``pyDatalog`` installation.

Overview
--------

Use the package through:

.. code-block:: python

   from aria.datalog import pyDatalog

The main runtime surface lives in ``pyDatalog.py`` and supports common Datalog
operations such as declaring terms, asserting and retracting facts, loading
rules, and querying derived relations.

Representative operations include:

.. code-block:: python

   from aria.datalog import pyDatalog

   pyDatalog.create_terms("X, Y, parent, ancestor")
   pyDatalog.assert_fact("parent", "alice", "bob")
   pyDatalog.assert_fact("parent", "bob", "carol")
   pyDatalog.load("ancestor(X,Y) <= parent(X,Y)")
   pyDatalog.load("ancestor(X,Y) <= parent(X,Z) & ancestor(Z,Y)")
   print(pyDatalog.ask("ancestor('alice',Y)"))

Package Structure
-----------------

Key files in ``aria/datalog`` include:

- ``pyDatalog.py``: public runtime API
- ``Logic.py``: logic-engine state management
- ``pyEngine.py``: evaluation engine internals
- ``pyParser.py``: parsing and query handling
- ``Aggregate.py``: aggregate support
- ``util.py``: utilities and exceptions
- ``grammar.txt``: grammar reference

Status and Provenance
---------------------

The top-level package README documents the intended status of this directory:

- it is a flattened vendored copy of ``pyDatalog``;
- the upstream project is unmaintained;
- ARIA keeps only the runtime-oriented pieces needed locally;
- a small Python 3 compatibility adjustment replaces
  ``inspect.getargspec()`` with ``inspect.signature()``.

Examples
--------

The repository includes runnable examples under ``aria/datalog/examples``.
From the repo root, you can run:

.. code-block:: bash

   python aria/datalog/examples/tutorial_example.py
   python aria/datalog/examples/datalog_example.py
   python aria/datalog/examples/graph_example.py
   python aria/datalog/examples/queens_example.py
   python aria/datalog/examples/python_objects_example.py

These cover recursive family relations, employee-style Datalog facts and
aggregates, graph reachability, the eight-queens problem, and querying Python
objects through Datalog rules.

Further Reading
---------------

- ``aria/datalog/README.md`` for ARIA-specific packaging notes
- ``aria/datalog/examples/README.md`` for the example map
- ``aria/datalog/UPSTREAM_README.md`` for upstream usage context
- ``aria/datalog/UPSTREAM_LICENSE`` for licensing information
