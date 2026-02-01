# Unification

Logic variable unification and pattern matching for Python.

This package provides:

- **Logic variables** (`Var`) that unify with any term and support reification.
- **Unification** (`unify`) and **reification** (`reify`) over Python objects, dicts, and sequences.
- **Multiple dispatch** for extensible `unify`/`reify` behaviour via the `dispatch` namespace.
- **Pattern matching** via `match.Dispatcher`, which selects implementations by unifying argument shapes.

## Usage

```python
from aria.unification import unify, reify, var

x, y = var("x"), var("y")
s = unify([1, x], [1, 2])   # {~x: 2}
reify([1, x], s)            # [1, 2]
```

See the package docstring and tests for more examples.
