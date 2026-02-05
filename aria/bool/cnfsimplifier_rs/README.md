# cnfsimplifier-rs

Rust re-implementation of the CNF simplifier used by `aria.bool.cnfsimplifier`, with Python bindings via PyO3.

## Features

- Same simplification algorithms as the Python implementation (tautology, subsumption, blocked clause, hidden and asymmetric variants).
- `simplify_numeric_clauses()` is used automatically by `aria.bool.simplify_numeric_clauses` when this extension is installed.
- All elimination functions are exposed to Python for direct use on numeric clauses (`List[List[int]]`).

## Build

Requires [Rust](https://rustup.rs/) and [maturin](https://github.com/PyO3/maturin).

```bash
# From the crate directory
cd aria/bool/cnfsimplifier_rs
maturin develop
```

Or from the project root:

```bash
pip install maturin
cd aria/bool/cnfsimplifier_rs && maturin develop
```

## Usage

Once built and installed, the Rust backend is used automatically for `simplify_numeric_clauses`:

```python
from aria.bool import simplify_numeric_clauses
from aria.bool.cnfsimplifier import rust_backend_available

if rust_backend_available():
    print("Using Rust backend")
clauses = [[1], [1, 2], [2, 3]]
simplified = simplify_numeric_clauses(clauses)  # [[1], [2, 3]]
```

Direct use of the Rust extension (all take/return `List[List[int]]`):

```python
import cnfsimplifier_rs

clauses = [[1, -1], [2, 3], [1, -2]]
cnfsimplifier_rs.cnf_tautology_elimination(clauses)  # drop [1, -1]
cnfsimplifier_rs.cnf_subsumption_elimination(clauses)
# ... cnf_blocked_clause_elimination, cnf_hidden_* , cnf_asymmetric_* ...
```

Or via the Python wrapper (raises if the extension is not installed):

```python
from aria.bool.cnfsimplifier.rust_backend import (
    is_available,
    simplify_numeric_clauses,
    cnf_tautology_elimination,
    cnf_subsumption_elimination,
    # ... etc.
)
```

## Optional dependency

To declare the Rust extension as an optional dependency of aria (e.g. for packaging):

```toml
[project.optional-dependencies]
cnf-rs = ["cnfsimplifier-rs"]
```

Then: `pip install aria[cnf-rs]` after building and publishing the `cnfsimplifier-rs` wheel, or use `maturin develop` locally.
