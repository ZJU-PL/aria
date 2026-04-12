# Datalog

This directory contains the flattened vendored `pyDatalog` runtime for ARIA.

- Import path: `from aria.datalog import pyDatalog`
- New Pythonic API: `from aria.datalog import Program`
- Examples:
  [aria/datalog/examples](/Users/rainoftime/Work/logic/aria/aria/datalog/examples)
- Upstream status: unmaintained
- Upstream README:
  [UPSTREAM_README.md](/Users/rainoftime/Work/logic/aria/aria/datalog/UPSTREAM_README.md)
- Upstream license:
  [UPSTREAM_LICENSE](/Users/rainoftime/Work/logic/aria/aria/datalog/UPSTREAM_LICENSE)

Migration notes:

- The package lives under the ARIA namespace instead of as a top-level install.
- A small Python 3 compatibility fix replaces `inspect.getargspec()` with
  `inspect.signature()`.
- Only the runtime-oriented upstream files needed by ARIA were kept here; extra
  upstream documentation/reference artifacts were removed.

## Pythonic API

The original vendored surface is still available, but ARIA now also provides a
more explicit builder-style API inspired by the workflow of Z3's Python APIs:
declare relations, assert facts and rules, then issue queries.

```python
from aria.datalog import Program

p = Program()
parent = p.relation("parent", 2)
ancestor = p.relation("ancestor", 2)
X, Y, Z = p.vars("X Y Z")

p.fact(parent("bill", "john"))
p.fact(parent("john", "sam"))

p.rule(ancestor(X, Y)).when(parent(X, Y))
p.rule(ancestor(X, Y)).when(parent(X, Z), ancestor(Z, Y))

result = p.query(ancestor("bill", Y))
print(result.rows())       # [('john',), ('sam',)]
print(result.named_rows()) # [{'Y': 'john'}, {'Y': 'sam'}]
print(result.scalar_rows()) # ['john', 'sam']
print(result.first_value()) # 'john'
```

For short scripts, the `Program` object also exposes convenience helpers for
bulk fact management and common query patterns:

```python
from aria.datalog import Program

p = Program()
parent = p.relation("parent", 2)
ancestor = p.relation("ancestor", 2)
X, Y, Z = p.vars("X Y Z")

p.facts(parent("bill", "john"), parent("john", "sam"))
p.rule_of(ancestor(X, Y), parent(X, Y))
p.rule_of(ancestor(X, Y), parent(X, Z), ancestor(Z, Y))

assert p.exists(ancestor("bill", Y))
print(p.scalar_rows(ancestor("bill", Y)))  # ['john', 'sam']
print(p.one_value(parent("bill", Y)))      # 'john'
```

Function-style predicates and aggregation heads are now also available through
the Pythonic layer:

```python
from aria.datalog import Program

p = Program()
manager = p.function("manager", 1)
salary = p.function("salary", 1)
budget = p.function("budget", 1)
indirect_manager = p.relation("indirect_manager", 2)
X, Y, Z, N = p.vars("X Y Z N")

p.define_of(manager("sam") == "mary")
p.define_of(manager("john") == "mary")
p.define_of(salary("sam") == 5900)
p.define_of(salary("john") == 6100)

p.rule_of(indirect_manager(X, Y), manager(X) == Y)
p.rule_of(indirect_manager(X, Y), manager(X) == Z, indirect_manager(Z, Y))
p.define_of(
    budget(X) == p.agg.sum(N, for_each=Y),
    indirect_manager(Y, X),
    salary(Y) == N,
)

print(p.one_value(budget("mary") == N))  # 12000
```

String-based loading is also supported for authoring rules in the original
`pyDatalog` syntax:

```python
from aria.datalog import Program

p = Program()
ancestor = p.relation("ancestor", 2)
Y = p.var("Y")

p.load(
    """
    + parent('bill', 'john')
    + parent('john', 'sam')
    ancestor(X, Y) <= parent(X, Y)
    ancestor(X, Y) <= parent(X, Z) & ancestor(Z, Y)
    """
)

print(p.query(ancestor("bill", Y)).scalar_rows())  # ['john', 'sam']
```

If the rules live in a `.dl` or `.py`-style source file, use
`Program.load_file(...)`:

```python
from aria.datalog import Program

p = Program()
p.load_file("rules/family.dl")
```

Textual queries are also supported, which is useful when both rules and queries
come from external sources:

```python
from aria.datalog import Program

p = Program()
p.load_file("rules/family.dl")
result = p.ask("ancestor('bill', Y)")
print(result.scalar_rows())
```

Text parsing errors from `load(...)`, `load_file(...)`, and `ask(...)` are
surfaced as `DatalogParseError` with line and column information:

```python
from aria.datalog import DatalogParseError, Program

try:
    Program().ask("ancestor('bill', Y")
except DatalogParseError as exc:
    print(exc.line, exc.column)
```

Design notes:

- This is a thin wrapper over the vendored `pyDatalog` engine, not a rewrite.
- Each `Program` now snapshots and restores the vendored thread-local engine
  state around every operation, so multiple `Program` instances can coexist in
  one thread without clobbering each other's facts and rules.
- Advanced users can keep using `pyDatalog` directly for aggregates, mixins, and
  lower-level engine access.
- Querying an undeclared predicate now raises `UndefinedPredicateError` instead
  of silently returning an empty result. A declared relation with no facts or
  rules still returns an empty result, which is the expected Datalog behavior.
- `Program.load(...)` pre-registers relations it discovers in the source string
  before delegating to the vendored parser, so string-loaded rules remain
  queryable via the explicit API.
- `Program.load_file(...)` is a thin convenience wrapper over `Program.load(...)`
  for loading external rule files.
- `Program.ask(...)` executes a query string and materializes it as the same
  `QueryResult` used by the builder API.
- `Program.facts(...)`, `retract_all(...)`, and `rule_of(...)` cover the common
  multi-statement cases without dropping down to loops.
- `Program.function(...)`, `define(...)`, and `define_of(...)` expose the
  upstream function-style head syntax without leaving the builder API.
- `Program.agg` exposes aggregate builders such as `count`, `sum`, `min`,
  `max`, `tuple`, `concat`, `rank`, and `running_sum`.
- `Program.exists(...)`, `rows(...)`, `scalar_rows(...)`, `first(...)`,
  `first_value(...)`, `one(...)`, and `one_value(...)` make the common query
  access patterns available directly on `Program`.
- Text parsing failures now raise `DatalogParseError` with source name, line,
  column, and source-line context.

## Result ergonomics

`QueryResult` supports several convenience accessors:

- `rows()`: list of tuples
- `named_rows()`: list of dicts keyed by variable name
- `scalar_rows()`: flatten single-column queries
- `first()` / `first_value()`: first row or first scalar
- `one()` / `one_value()`: require exactly one row
- iteration and `len(result)`

## Gaps

Current limitations of the Pythonic layer:

- Mixin-backed object queries still use the lower-level `pyDatalog` surface.
- The Pythonic layer now covers common relation, function, aggregate, and
  text-loading workflows, but it still does not wrap every upstream extension.
