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

The main runtime surface lives in ``py_datalog.py`` and supports common Datalog
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

The module files use snake_case names, while the package exposes the historical
pyDatalog ``CamelCase`` spellings as package attributes (for example
``from aria.datalog import pyParser`` returns the ``py_parser`` module). The
attributes are aliases, not separate importable submodules, so
``import aria.datalog.pyParser`` does not work.

Key files in ``aria/datalog`` include:

- ``py_datalog.py``: public runtime API (also available as the package
  attribute ``aria.datalog.pyDatalog``)
- ``logic.py``: logic-engine state management (its ``Logic`` class is
  re-exported at package level and the module is also available as
  ``aria.datalog.logic``)
- ``py_engine.py``: evaluation engine internals (also available as
  ``aria.datalog.pyEngine``)
- ``py_parser.py``: parsing and query handling (also available as
  ``aria.datalog.pyParser``)
- ``aggregate.py``: aggregate support (the package attribute ``Aggregate`` is
  this module)
- ``api.py``: object-oriented ``Program`` / ``Relation`` / ``QueryResult`` API
- ``user_list.py``: ``UserList`` sequence type (the package attribute
  ``UserList`` is this module)
- ``util.py``: utilities and exceptions
- ``version.py``: upstream package version string (``0.17.4``)
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
