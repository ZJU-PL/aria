# Native Python Concolic Testing

`aria.concolic` combines native Python path tracing with Aria's constrained
solution samplers. It executes the target with ordinary Python values while a
separate shadow environment constructs and validates Z3 expressions. It does
not depend on Py-Conbyte or CrossHair.

## Safety and soundness

Instrumentation preserves concrete evaluation order: a target expression is
evaluated once, then its already-computed value is passed to a runtime hook.
Before a symbolic assignment or branch is used for exploration, the backend
checks that it agrees with the concrete execution under the current inputs.
An inconsistent model is discarded and reported as a structured
`symbolic-concrete-mismatch` diagnostic.

Unsupported symbolic operations do not stop concrete execution and are never
silently approximated. The trace contains an explicit diagnostic and the
affected predicate is excluded from frontier generation.

Production campaigns should use worker isolation, which is the CLI default.
Each input runs in a spawned process with a hard wall-clock limit. Direct mode
is useful for debugging and for non-importable local functions.
Worker stdout and stderr are captured with a fixed upper bound. On platforms
with `resource.RLIMIT_AS`, campaigns may also impose an address-space limit.
Worker isolation is a reliability boundary, not a security sandbox for
untrusted Python code.

## Python API

```python
from aria.concolic import ConcolicEngine, ConcolicOptions


def classify(value: int, prefix: str) -> str:
    text = prefix + str(value)  # `str(int)` currently becomes opaque
    if value >= 10 and prefix.startswith("ab"):
        return "interesting"
    return text


engine = ConcolicEngine(
    classify,
    ConcolicOptions(
        isolate=True,
        execution_timeout=5.0,
        max_iterations=100,
        samples_per_frontier=8,
    ),
)
result = engine.explore([{"value": 0, "prefix": ""}])
```

Use `await engine.atrace(...)` and `await engine.aexplore(...)` for asynchronous
targets. Generator and async-generator targets are driven under the active
trace context and bounded by `max_yields`.

## CLI

```bash
aria-concolic mypackage.parser:parse \
  --seed '{"text":"12","base":10}' \
  --max-iterations 200 \
  --samples-per-frontier 8 \
  --timeout 5 \
  --solver-timeout 3 \
  --memory-limit-mb 1024 \
  --artifact .aria-concolic/parser.json \
  --pytest-output tests/test_concolic_regressions.py \
  --pretty
```

The command exits with `0` when no exception is discovered, `1` when the
campaign discovers an exception, and `2` for configuration or instrumentation
errors.

Target execution and frontier solving have independent deadlines. Sampler
failures and exact-Z3 fallback events are retained as campaign diagnostics in
the JSON result rather than being hidden in logs.

## Compatibility contract

Symbolically supported inputs:

- exact `bool`, arbitrary-precision `int`, and `str` values
- Latin-1-preserving ``bytes``, mutable ``bytearray``, finite ``set`` values,
  and IEEE-754 binary64 ``float`` inputs
- fixed-shape nested `list`, `tuple`, and `dict` inputs whose leaves are
  supported primitive values; leaves are sampled independently and rebuilt
  into the original container types
- bounded concrete shape variants for lists, tuples, dictionaries, sets,
  bytes, and bytearrays with ``--expand-input-shapes``
- default primitive arguments are treated as constants unless included in the
  seed object

Symbolically supported expressions:

- integer `+`, `-`, `*`, Python-exact `//`, and Python-exact `%`
- Boolean operations, truthiness, chained comparisons, and conditional
  expressions
- string concatenation, length, indexing, containment, `startswith`,
  `endswith`, and `find`
- `abs()` and `bool()`

Instrumented control flow and state:

- simple and chained name assignments, annotated assignments, augmented
  assignments, and assignment expressions
- `if`, `elif`, `while`, `assert`, conditional expressions, and `for` over
  one-, two-, or three-argument `range`
- exception outcomes, synchronous functions, coroutines, generators, and
  async generators
- interprocedural calls and symbolic returns across allowlisted package
  functions, static methods, and class methods
- list ``append``, ``extend``, ``insert``, ``pop``, ``clear``, indexed
  assignment and deletion; dictionary assignment, ``update``, ``setdefault``,
  ``pop``, and ``clear``
- dataclass and public object fields, including direct and augmented attribute
  assignment

The following currently execute concretely but lose symbolic precision with
an explicit diagnostic when their value reaches a branch:

- arbitrary custom mutable containers, slices, complex descriptors, and
  floating-point operations beyond the modeled arithmetic/comparisons
- arbitrary calls and unmodelled standard-library functions
- bitwise operations and exponentiation
- non-`range` iteration and destructuring assignment
- mutually recursive calls outside selected instrumentation packages, C
  extension internals, and frame-sensitive calls such as ``eval``/``exec``

Isolated execution requires an importable module-level function. Bound methods
and local functions can be traced only in direct mode. Functions containing
`global` or `nonlocal` declarations are rejected because recompiling them
outside their original closure would not preserve semantics.

## Artifacts and replay

`ArtifactStore` writes the versioned `aria.concolic/v1` JSON schema. Artifacts
contain inputs, outcomes, stable source-derived branch identifiers, path
signatures, diagnostics, coverage, and campaign statistics. `ArtifactStore.replay`
checks that outcome, path, and exception information remain stable.

`engine.shrink(failing_trace)` minimizes primitive inputs with Z3 Optimize and
accepts the candidate only after it reproduces the same exception type and
message. Generated pytest regressions call the original target with the
recorded inputs.

Use `ArtifactStore.areplay(...)` and `await engine.ashrink(...)` in async
applications. Generated regression tests share the engine's signature-aware
invocation helper, including positional-only and keyword-only arguments.

## Symbolic function models

Built-ins and string methods are resolved through `SymbolicModelRegistry`.
Applications can extend the process-wide default registry:

```python
import z3
from aria.concolic import SymbolicTerm, register_function_model


def clamp_model(args):
    value = args[0]
    if value.expression is None:
        return None
    return SymbolicTerm(
        z3.If(value.expression >= 0, value.expression, 0),
        value.guards,
    )


register_function_model("clamp_positive", clamp_model)
```

Registrations are process-local. For spawn-isolated campaigns, register custom
models while importing the target module so every worker installs them.

Default guarded model packs cover:

- parsing: decimal ``int(str)`` and non-negative ``str(int)``;
- text: ``ord`` and ``chr``;
- URL/string processing: guarded ``strip``/``lstrip``, exact one-occurrence
  ``replace``, bounded ``find``, ASCII predicates, lowercase identity regions,
  slicing, and one-split tuple propagation;
- collections: ``sum``, ``min``, ``max``, ``all``, and ``any``;
- numeric: non-negative constant-exponent ``pow``;
- paths: simple POSIX/NT ``join`` and slash-free ``basename`` cases.

Models add definedness guards and are discarded when the concrete input falls
outside their proven semantics.

## Interprocedural package instrumentation

```bash
aria-concolic mypackage.api:entry \
  --seed '{"value":0}' \
  --instrument-package mypackage \
  --instrument-omit 'mypackage.generated*' \
  --coverage --coverage-source mypackage
```

The package manager imports allowlisted modules, instruments source-available
functions and methods, rewires cross-module aliases, and propagates symbolic
arguments and returns. ``--instrument-include`` and ``--instrument-omit`` are
repeatable glob rules. ``--no-import-submodules`` restricts instrumentation to
modules already imported by the target.

Patches are applied only inside a locked, reentrant execution context and the
exact prior module/class descriptors are restored afterward. A guarded import
hook instruments allowlisted modules loaded lazily during execution after their
module initialization completes. This permits multiple engines to share a
package without leaving persistent monkey patches.

## Search scaling without solver parallelism

The engine bounds and prioritizes sequential exploration through:

- an LRU cache for satisfiable and unsatisfiable frontier queries;
- path-state subsumption with `max_inputs_per_path`;
- `max_path_depth`, `max_constraint_nodes`, and `max_frontiers` budgets;
- a campaign-wide wall-clock deadline;
- coverage ordering that prefers uncovered, rarer, smaller, shallower
  frontiers.

Search counters and sampler-cache statistics are included in
`ConcolicResult.search_stats` and campaign artifacts.

## Conventional coverage percentages

Enable coverage collection through `ConcolicOptions`:

```python
engine = ConcolicEngine(
    target,
    ConcolicOptions(
        measure_coverage=True,
        coverage_sources=("mypackage",),
    ),
)
result = engine.explore([{"value": 0}])

print(result.coverage.lines.percent)
print(result.coverage.statements.percent)
print(result.coverage.functions.percent)
print(result.coverage.branches.percent)
print(result.coverage.target_function.branches.percent)
```

Or use the CLI:

```bash
aria-concolic mypackage.module:target \
  --seed '{"value":0}' \
  --coverage \
  --coverage-source mypackage \
  --coverage-json coverage.json
```

Reports merge data from every direct execution or isolated worker. They include
aggregate and per-file covered/total counts, percentages, missing executable
lines, missing branch arcs, function coverage, and a separate summary for the
target function. In Python, coverage.py represents executable statements by
source line, so the reported line and statement percentages intentionally use
the same denominator.

Without `coverage_sources`, totals cover the target source file. Repeat
`--coverage-source` to calculate percentages across a package or several source
directories. Conventional coverage is included in the normal JSON result and
versioned campaign artifact as well as the optional standalone report.

Coverage thresholds and curves are available for CI:

```bash
aria-concolic mypackage.api:entry \
  --seed '{"value":0}' \
  --instrument-package mypackage \
  --coverage --coverage-source mypackage \
  --fail-under-lines 80 \
  --fail-under-functions 90 \
  --fail-under-branches 85 \
  --coverage-timeline-json coverage-timeline.json \
  --coverage-timeline-csv coverage-timeline.csv
```

A threshold failure exits with status 3. Coverage snapshots are cumulative for
each campaign iteration.

## Pytest, Hypothesis, and property oracles

``assert_concolic`` evaluates trace-level properties and raises a
``ConcolicAssertionError`` containing generated counterexamples. Use
``pytest_parameters(result)`` to parameterize regression tests and
``hypothesis_strategy(result)`` to feed concolic inputs into Hypothesis.
``assert_concolic_async`` supports coroutine targets.

## Benchmarking real libraries

Run the manifest-driven comparison suite from the repository root:

```bash
python scripts/run_concolic_benchmarks.py \
  --manifest benchmarks/concolic/manifest.json \
  --output-json /tmp/concolic-benchmarks.json \
  --output-csv /tmp/concolic-benchmarks.csv
```

Each case compares a seed-only baseline against a full campaign in fresh
processes and reports coverage, unique paths, failures, frontier queries, cache
hits, executions, and elapsed time.

## Explaining symbolic gaps

Use ``--explain-gaps`` to rank operations that caused symbolic precision loss:

```bash
aria-concolic mypackage.api:entry \
  --seed '{"value":0}' \
  --instrument-package mypackage \
  --explain-gaps --gaps-json symbolic-gaps.json
```

The report groups diagnostics by operation and source location, counts
occurrences and blocked frontiers, records representative input types/values,
and suggests package instrumentation, a model pack, or a semantics extension.
Opaque branches are also emitted as synthetic source-located hotspots even when
the originating expression diagnostic occurred outside a branch hook.
