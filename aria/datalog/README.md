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

- Aggregates, function-style predicates, and mixin-backed object queries still
  use the lower-level `pyDatalog` surface.
- The API currently focuses on relation-style predicates rather than the full
  function and aggregate feature set of upstream `pyDatalog`.
