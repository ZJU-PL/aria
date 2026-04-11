CLI Tools
=========

The ``aria.cli`` package provides command-line entrypoints for several user-facing
reasoning workflows. After ``pip install -e .``, these tools are exposed both as
console scripts such as ``aria-mc`` and as Python module entrypoints such as
``python -m aria.cli.mc_cli``.

Registered console scripts
--------------------------

The current ``pyproject.toml`` registers these commands:

.. list-table::
   :header-rows: 1

   * - Console command
     - Module entrypoint
     - Purpose
   * - ``aria-fmldoc``
     - ``python -m aria.cli.fmldoc_cli``
     - Format conversion, validation, and analysis
   * - ``aria-mc``
     - ``python -m aria.cli.mc_cli``
     - Model counting for Boolean, bit-vector, and arithmetic inputs
   * - ``aria-pyomt``
     - ``python -m aria.cli.pyomt_cli``
     - Optimization modulo theories and MaxSMT frontends
   * - ``aria-efsmt``
     - ``python -m aria.cli.efsmt_cli``
     - Exists-forall SMT solving
   * - ``aria-maxsat``
     - ``python -m aria.cli.maxsat_cli``
     - Weighted partial MaxSAT from WCNF input
   * - ``aria-unsat-core``
     - ``python -m aria.cli.unsat_core_cli``
     - UNSAT core, MUS, and MSS workflows
   * - ``aria-allsmt``
     - ``python -m aria.cli.allsmt_cli``
     - AllSMT enumeration
   * - ``aria-smt-server``
     - ``python -m aria.cli.smt_server_cli``
     - IPC-based SMT server
   * - ``aria-efmc``
     - ``python -m aria.cli.efmc_cli``
     - Transition-system verification
   * - ``aria-efmc-efsmt``
     - ``python -m aria.cli.efmc_efsmt_cli``
     - EFMC-specific EFSMT frontend
   * - ``aria-polyhorn``
     - ``python -m aria.cli.polyhorn_cli``
     - Polynomial Horn solving

Quick start
-----------

.. code-block:: bash

   python -m aria.cli.fmldoc_cli translate -i input.cnf -o output.smt2
   python -m aria.cli.mc_cli formula.smt2
   python -m aria.cli.pyomt_cli problem.smt2
   python -m aria.cli.efsmt_cli problem.smt2
   aria-maxsat formula.wcnf --solver rc2
   aria-unsat-core formula.smt2
   aria-allsmt formula.smt2 --limit 100
   python -m aria.cli.smt_server_cli
   python -m aria.cli.efmc_cli --help
   python -m aria.cli.polyhorn_cli --help

Tool summaries
--------------

``fmldoc``
~~~~~~~~~~

Translate, validate, and analyze supported logic-constraint files.

.. code-block:: bash

   python -m aria.cli.fmldoc_cli translate -i input.cnf -o output.smt2
   python -m aria.cli.fmldoc_cli validate -i input.smt2 -f smtlib2
   python -m aria.cli.fmldoc_cli analyze -i input.cnf
   python -m aria.cli.fmldoc_cli formats

``mc``
~~~~~~

Count satisfying models for DIMACS or SMT-LIB2 inputs.

.. code-block:: bash

   python -m aria.cli.mc_cli formula.smt2
   python -m aria.cli.mc_cli formula.cnf --theory bool
   python -m aria.cli.mc_cli formula.smt2 --theory bv
   python -m aria.cli.mc_cli formula.smt2 --theory arith

Key options: ``--theory``, ``--method``, ``--project``, ``--timeout``,
``--log-level``.

``pyomt``
~~~~~~~~~

Solve optimization problems from SMT-LIB2 input.

.. code-block:: bash

   python -m aria.cli.pyomt_cli problem.smt2
   python -m aria.cli.pyomt_cli problem.smt2 --engine qsmt
   python -m aria.cli.pyomt_cli problem.smt2 --engine iter
   python -m aria.cli.pyomt_cli problem.smt2 --engine maxsat

Key options: ``--type``, ``--theory``, ``--engine``, ``--solver``,
``--log-level``.

``efsmt``
~~~~~~~~~

Solve exists-forall SMT problems with multiple backends.

.. code-block:: bash

   python -m aria.cli.efsmt_cli problem.smt2
   python -m aria.cli.efsmt_cli problem.smt2 --parser z3
   python -m aria.cli.efsmt_cli problem.smt2 --theory bv
   python -m aria.cli.efsmt_cli problem.smt2 --engine efbv-par

Key options include ``--parser``, ``--theory``, ``--engine``, ``--timeout``,
``--max-loops``, plus theory-specific solver options for bit-vectors and
LIA/LRA.

``maxsat``
~~~~~~~~~~

Solve weighted partial MaxSAT problems from WCNF input.

.. code-block:: bash

   aria-maxsat formula.wcnf
   aria-maxsat formula.wcnf --solver rc2
   aria-maxsat formula.wcnf --print-model

``unsat_core``
~~~~~~~~~~~~~~

Compute one UNSAT core or enumerate MUSes from SMT-LIB2 input.

.. code-block:: bash

   aria-unsat-core formula.smt2
   aria-unsat-core formula.smt2 --algorithm musx
   aria-unsat-core formula.smt2 --all-mus

``allsmt``
~~~~~~~~~~

Enumerate satisfying models, optionally projected to selected variables.

.. code-block:: bash

   aria-allsmt formula.smt2
   aria-allsmt formula.smt2 --limit 50
   aria-allsmt formula.smt2 --project x,y,z
   aria-allsmt formula.smt2 --count-only

``smt_server``
~~~~~~~~~~~~~~

Run an IPC-based SMT server with commands such as ``assert``, ``check-sat``,
``get-model``, ``allsmt``, ``unsat-core``, and ``backbone``.

.. code-block:: bash

   python -m aria.cli.smt_server_cli
   python -m aria.cli.smt_server_cli --input-pipe /tmp/in --output-pipe /tmp/out

``efmc`` / ``efmc_efsmt`` / ``polyhorn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These entrypoints expose the verification and quantified-reasoning stacks:

* ``aria-efmc`` for verification over CHC, SyGuS, Boogie, and C-style inputs
* ``aria-efmc-efsmt`` for EFMC-focused EFSMT workflows
* ``aria-polyhorn`` for polynomial Horn solving

Validation
----------

For the current command surface, see:

* ``aria/cli/README.md``
* ``pyproject.toml`` under ``[project.scripts]``
* the parser definitions in ``aria/cli/*_cli.py``
