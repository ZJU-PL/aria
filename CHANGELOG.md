# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-21

### Added

- New modules: `aria.quant` (quantified reasoning with EFSMT, CHC, quantifier elimination), `aria.scsat`, `aria.symabs`
- Sampling infrastructure: QF_DTLIA, QF_SLIA, QF_FP, finite domain samplers
- Program verification: EFMC (PDR, SMTTools), bounded model checking for BTOR2
- LLM tools: `aria.llmtools` with induction, trigger selection, SMTO, codex support
- Datalog engine with Souffle integration and Pythonic APIs
- CFL reachability toolkit and automata tools
- Symbolic finite automata (SFA) module
- Finite field solver with automatic backend selection
- Probability/WMI module: core, boolean, arithmetic subpackages
- Translator registry (OPB, QCIR, DIMACS WCNF, etc.)
- New CLI tools: `aria-efsmt`, `aria-efmc`, `aria-polyhorn`, `aria-pypmt`, `aria-allsmt`
- Optional Rust backend for CNF simplification
- Type stubs and developer guidelines (AGENTS.md)

### Changed

- Renamed `aria.optimization` → `aria.pyomt`
- Renamed `aria.efsyn` → `aria.efdual`
- Renamed `aria.monabs` → `aria.scsat`
- Reorganized CLI tools into individual modules
- Upgraded PySMT and PySAT dependencies
- Improved parallel utilities and async worker management
- Better test coverage across modules

### Fixed

- Concurrency bugs in parallel utilities
- EFSMT worker stability issues
- Import errors and type consistency
- MaxSAT anytime solver binary search → core-guided approach
- Various pylint warnings and code quality issues

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

### Added

- **CLI**: `aria-maxsat` — MaxSAT solver for WCNF (engines: RC2, FM, LSU)
- **CLI**: `aria-unsat-core` — UNSAT core / MUS / MSS from SMT-LIB2 (marco, musx, optux)
- **CLI**: `aria-allsmt` — Enumerate all satisfying models of SMT formulas
- **CLI**: `aria-pypmt` — Planning modulo theories (PDDL-based planning as SMT)
- **New module**: `aria.ml.gansat` — Generative adversarial network-based SMT sampling
- **New module**: `aria.itp.fstar-copilot` — F* copilot agent with proof debugging, verification, and project setup skills
- **New module**: `aria.quant.efdual` — Dual-memory CEGIS solvers for exists-forall synthesis (renamed from `efsyn`)
- **New module**: `aria.sampling.dtlia` — Constrained sampling for QF_UFDTLIA
- **New module**: `aria.volumn` — Volume computation for SMT (LRA)
- **SRK**: Proof-based interpolation support, UltPeriodic sequences, VAS/VASS helpers, LLRF residual computation, De Bruijn expression types

### Changed

- **Renamed**: `aria.monabs` → `aria.scsat` (shared-context batched satisfiability)
- **Renamed**: `aria.quant.efsyn` → `aria.quant.efdual` (exists-forall dual-memory CEGIS)
- **Refined**: `aria.quant.efdual` CEGIS search semantics and memory management
- **Improved**: `aria.bool.knowledge_compile` — knowledge compilation improvements
- **Improved**: `aria.pyomt` — OMT/OMTFP solver improvements
- **Improved**: EFSMT worker reuse, cleanup, and profiling stability
- **Reorganized**: `aria.srk` — aligned with OCaml version, improved utilities
- **Reorganized**: Project structure with cleaner package layout

### Fixed

- `aria.sampling.dtlia`: QF_UFDTLIA sampler recursion limit and true int bounds
- Import errors in SRK modules

### Documentation

- Updated docs for `monabs` → `scsat` rename across all RST/MD files
- Added `aria-pypmt` to CLI documentation
- Fixed `efdual` README title reference

### Planned

- Additional documentation and tutorials
- More example applications
- Performance optimizations
- Extended solver support

