# AGENTS.md - Developer Guidelines for ARIA

This repository is a broad automated reasoning monorepo, not a single
uniform library. It mixes maintained library code, CLI entrypoints, research
prototypes, benchmark corpora, solver wrappers, and experiment scripts.

Use this file for repository-wide guidance. Before editing inside a subsystem,
also check the nearest local `README.md` and any deeper `AGENTS.md`.

## Repo Shape

Top-level areas you are likely to touch:

- `aria/bool`: SAT, MaxSAT, QBF, CNF simplification, NNF, knowledge compilation
- `aria/smt`: SMT-related reasoning utilities
- `aria/counting`: model counting across Boolean and SMT fragments
- `aria/optimization`: OMT / MaxSMT code
- `aria/quant`: quantified reasoning and EFSMT experiments
- `aria/efmc`: verification and invariant-generation stack
- `aria/symabs`, `aria/monabs`: symbolic abstraction and program analysis
- `aria/srk`, `aria/itp`, `aria/fol`: symbolic reasoning / theorem-proving style code
- `aria/llmtools`, `aria/ml`: LLM integrations and ML-related experiments
- `aria/cli`: user-facing CLI tools installed via `pyproject.toml`
- `scripts/`: evaluation, SMT-COMP, solver, and one-off research scripts
- `benchmarks/`: benchmark inputs; do not "clean up" or reformat casually

Several subpackages have their own `README.md`. Read the local one before
making nontrivial changes so you understand whether you are in production code,
port code, or a research artifact.

## Environment and Tooling

Primary project configuration lives in `pyproject.toml`.

- Python support in packaging: 3.8+
- Mypy is configured with `python_version = 3.9`
- Formatting uses Black with line length 88 and isort with `profile = "black"`
- Pytest configuration is in `pyproject.toml`
- Pylint uses `.pylintrc`

Typical setup:

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Alternative local setup script:

```bash
bash setup_local_env.sh
```

Notes:

- `README.md` documents a `.venv` workflow, while `setup_local_env.sh` creates
  `venv/`. Do not assume both are present.
- Some tests and scripts depend on optional external solvers or binaries.
  Treat missing-environment failures differently from logic regressions.

## Build, Lint, and Test Commands

General commands:

```bash
pytest
pytest --cov=aria
pytest -m "not slow"

mypy aria/
pylint aria/

black aria/
isort aria/
```

Prefer targeted tests for the subsystem you changed:

```bash
pytest aria/tests/test_bool_engines.py
pytest aria/efmc/tests/test_cli_efmc.py
pytest aria/quant/efbv/tests/test_efbv_parallel.py
pytest aria/unification/tests
```

For CLI-facing changes, prefer running the specific module entrypoint:

```bash
python -m aria.cli.fmldoc --help
python -m aria.cli.mc --help
python -m aria.cli.pyomt --help
python -m aria.cli.efsmt --help
python -m aria.cli.maxsat --help
python -m aria.cli.unsat_core --help
python -m aria.cli.allsmt --help
python -m aria.cli.smt_server --help
python -m aria.efmc.cli.efmc --help
```

## Coding Conventions

- Keep line length within 88 characters.
- Add type hints to new or modified code.
- Because mypy targets Python 3.9, prefer `Optional[T]`, `List[T]`, `Dict[K, V]`,
  and `Tuple[...]` over Python 3.10+ syntax such as `T | None` or `list[T]`
  unless the surrounding file already consistently uses newer syntax and the
  change is intentionally local.
- Group imports as standard library, third-party, then local imports.
- Use `CamelCase` for classes and `snake_case` for functions and methods.
- Preserve existing naming in ported or math-heavy code when a cleanup would
  reduce correspondence with papers or upstream implementations.

## Tests

Test conventions are mixed across the repo.

- Some tests use `from aria.tests import TestCase, main`
- Some use `from aria.efmc.tests import TestCase, main`
- Many subsystems use plain `unittest`
- Some newer tests use `pytest` features such as parametrization and markers

Do not rewrite tests to a different framework unless there is a clear reason.
Follow the style already used in the directory you are editing.

When adding tests:

- Put them in the nearest existing test directory for that subsystem
- Prefer focused regression tests over broad end-to-end suites
- Mark environment-sensitive tests carefully if they require optional solvers,
  network access, or heavyweight external tools

## Working in Research Code

Many areas in this repository are research implementations or ports. Expect:

- uneven abstraction quality
- duplicated concepts across subsystems
- solver-specific code paths
- partially implemented features
- comments referring to papers, experiments, or upstream artifacts

In these areas:

- preserve behavior over stylistic cleanup
- avoid "simplifying" algorithms unless you understand the proof/search impact
- keep paper terminology and solver terminology aligned with existing code
- document assumptions when changing heuristics, encodings, or search loops

## Benchmarks, Scripts, and Generated Inputs

- Do not mass-reformat files under `benchmarks/`
- Do not change benchmark semantics to satisfy a test
- Keep scripts runnable from the repo root unless the script already assumes
  another working directory
- When changing experiment scripts, preserve command-line behavior unless the task explicitly calls for an interface change

## Subsystem Guidance

Additional local instructions live in:

- `aria/efmc/AGENTS.md`
- `aria/quant/AGENTS.md`
- `aria/llmtools/AGENTS.md`
- `scripts/AGENTS.md`

Read the closest applicable file before making significant edits there.
