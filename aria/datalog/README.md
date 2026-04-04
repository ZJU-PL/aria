# Datalog

This directory contains the flattened vendored `pyDatalog` runtime for ARIA.

- Import path: `from aria.datalog import pyDatalog`
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
