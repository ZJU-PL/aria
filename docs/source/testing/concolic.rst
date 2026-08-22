Native Python concolic testing
==============================

``aria.concolic`` executes Python functions concretely while maintaining a
validated symbolic shadow state. Alternative branch constraints are passed to
Aria's theory-aware samplers to generate several candidate inputs per
frontier.

The backend is native to Aria and has no Py-Conbyte dependency. Its main
properties are:

* source-level AST instrumentation with stable branch identifiers;
* ordinary Python concrete values and single evaluation of target expressions;
* concrete-versus-symbolic validation before exploration;
* explicit diagnostics instead of silent symbolic fallbacks;
* LIA, NIA, Boolean, and SLIA sampling through ``aria.sampling``;
* spawn-isolated workers, wall-clock limits, and deterministic cleanup;
* synchronous, asynchronous, generator, and range-loop tracing;
* versioned artifacts, replay, shrinking, and pytest regression generation.
* symbolic lists, tuples, dictionaries, sets, bytes, bytearrays, binary64
  floating point, and bounded shape variants;
* pluggable symbolic function and method models;
* sequential query caching, path subsumption, and bounded priority search.
* conventional line, statement, function, and branch percentages through
  ``coverage.py``, including merged isolated-worker data.
* allowlisted interprocedural package instrumentation and symbolic returns;
* mutable containers, dataclasses, and stateful public object fields;
* guarded parsing, text, collection, numeric, and path model packs;
* pytest property assertions and Hypothesis input strategies;
* CI thresholds, cumulative coverage curves, and benchmark comparisons.

Coverage measurement
--------------------

Pass ``measure_coverage=True`` to ``ConcolicOptions`` or use the CLI flags
``--coverage`` and ``--coverage-json``. By default, totals describe the target
source file. ``--coverage-source PACKAGE_OR_DIRECTORY`` expands the denominator
to a real package or source tree. Reports contain aggregate, per-file, and
target-function metrics plus missing lines and branch arcs.

Package testing
---------------

Use ``--instrument-package`` with repeatable include/omit globs to instrument
source-available functions and methods across a package. Symbolic arguments and
returns are propagated across module aliases and isolated workers. Pair it with
``--coverage-source`` to calculate package-wide conventional percentages.
Package patches are reentrant and restored after each execution; lazily imported
allowlisted modules are instrumented after initialization.

CI and benchmarks
-----------------

``--fail-under-lines``, ``--fail-under-statements``,
``--fail-under-functions``, and ``--fail-under-branches`` enforce aggregate
coverage gates. Timeline JSON/CSV files show cumulative coverage by iteration.
The manifest under ``benchmarks/concolic`` compares seed-only and full campaigns
on multi-module and standard-library targets.

Gap explanations
----------------

``--explain-gaps`` ranks unsupported operations and blocked frontiers by source
location and frequency. ``--gaps-json`` exports the same report with input type
examples and suggested remediation.

See ``aria/concolic/README.md`` in the source distribution for the complete
API, CLI examples, and compatibility contract.
