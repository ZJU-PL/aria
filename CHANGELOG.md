# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-02

### Added

- Initial PyPI release
- Core automated reasoning components:
  - **srk**: Symbolic reasoning kernel
  - **smt**: SMT operations and utilities
  - **bool**: Boolean operations and engines
  - **quant**: Quantifier reasoning and solvers
  - **optimization**: Optimization and MaxSAT solvers
- Specialized modules:
  - Model counting and approximation
  - AllSMT (enumerate all satisfying models)
  - UNSAT core computation
  - Backbone literal computation
  - Abductive reasoning
  - Program synthesis
  - Interpolant generation
  - Symbolic abstraction
- Command-line tools:
  - `aria-smt-server`: Enhanced SMT server with SMT-LIB2 interface
  - `aria-pyomt`: Portfolio model testing
  - `aria-mc`: Model counting
  - `aria-efsmt`: Efficient functional SMT
  - `aria-fmldoc`: Formal documentation generation
- Comprehensive test suite
- Type stubs for external dependencies

### Changed

- Migrated from `py-arlib` to `aria` package name
- Updated to use `pyproject.toml` for modern Python packaging

### Dependencies

- PySMT==0.9.0
- z3-solver==4.12.0
- python-sat==0.1.8.dev1
- pyapproxmc==4.1.24
- numpy, lark, hypothesis, and more

### Known Issues

- Python 3.13 compatibility requires six.moves workaround

## [Unreleased]

### Planned

- Additional documentation and tutorials
- More example applications
- Performance optimizations
- Extended solver support

