# Global Parameters

Global configuration and path management for the Aria project.

## Components

- `__init__.py`: Main module exports
- `paths.py`: Path configuration and solver location management

## Features

- Centralized configuration for all Aria components
- Solver path management (Z3, CVC5, etc.)
- Project root and benchmark path references
- Environment-based configuration

## Usage

```python
from aria.global_params import (
    global_config,
    SMT_SOLVERS_PATH,
    PROJECT_ROOT,
    BIN_SOLVERS_PATH,
    BENCHMARKS_PATH
)

# Access configuration
z3_path = global_config.z3_path
```
