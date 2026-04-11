Ranking Functions
=================

Ranking functions are used to prove progress and termination-like properties in
verification workflows. In ARIA, ranking-function-related logic appears in EFMC
engines and supporting verification code, but earlier examples in this page had
drifted away from the current package surface.

Current status
--------------

The current EF engine surface is organized under ``aria.efmc.engines.ef``
around ``EFProver`` and related template-driven proving code.

How to approach ranking-related workflows today
-----------------------------------------------

* use the main verifier CLI for supported termination/progress-oriented flows
* inspect the EFMC engine code under ``aria/efmc/engines/ef/`` for the current
  template and proving implementation
* treat older ranking-template examples with caution unless they are backed by
  current code in the repository

CLI entrypoint
--------------

.. code-block:: bash

   aria-efmc --help

Notes
-----

This page remains a conceptual placeholder for ranking-function-oriented
verification and now points readers to the current EF engine layout instead of
stale module examples.
