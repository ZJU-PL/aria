# AGENTS.md - Developer Guidelines for ARIA

ARIA is a mixed automated-reasoning monorepo: libraries, CLI tools, research
prototypes, solver wrappers, scripts, and benchmark corpora.

Use this file for repo-wide guidance. Before nontrivial edits, also read the
nearest local `README.md` and any deeper `AGENTS.md`.

- `aria/bool`: SAT, MaxSAT, QBF, CNF simplification, NNF, KC
- `aria/smt`: SMT utilities
- `aria/counting`: model counting across Boolean and SMT fragments
- `aria/optimization`: OMT / MaxSMT
- `aria/quant`: quantified reasoning and EFSMT experiments
- `aria/efmc`: verification and invariant generation
- `aria/symabs`, `aria/monabs`: symbolic abstraction and program analysis
- `aria/srk`, `aria/itp`, `aria/fol`: symbolic reasoning / theorem proving
- `aria/llmtools`, `aria/ml`: LLM and ML work
- `aria/cli`: user-facing CLI entrypoints
- `scripts/`: evaluation, solver, and experiment scripts
- `benchmarks/`: benchmark inputs; do not casually reformat or clean up

Read local docs before changing production code, ports, or research artifacts.

## Environment

- Project config: `pyproject.toml`
- Python packaging target: 3.8+
- Mypy target: Python 3.9
- Formatting: Black (88 cols) and isort (`profile = "black"`)
- Lint/test config: `pyproject.toml`, `.pylintrc`
- Setup: `uv venv && source .venv/bin/activate && uv pip install -e .`
- Alternative setup: `bash setup_local_env.sh`
- `README.md` may assume `.venv`, while the setup script may create `venv/`
- Some tests need optional external solvers/binaries; separate env failures
  from logic regressions

## Build, Lint, Test

- General: `pytest`, `pytest --cov=aria`, `pytest -m "not slow"`
- Type/lint: `mypy aria/`, `pylint aria/`
- Format: `black aria/`, `isort aria/`
- Prefer targeted tests for the subsystem you changed
- For CLI changes, run the relevant module entrypoint with `--help`
- `pytest aria/tests/test_bool_engines.py`
- `pytest aria/efmc/tests/test_cli_efmc.py`
- `pytest aria/quant/efbv/tests/test_efbv_parallel.py`
- `python -m aria.cli.maxsat --help`
- `python -m aria.efmc.cli.efmc --help`

## Coding Guidelines

- Keep line length within 88 characters
- Add type hints to new or modified code
- Prefer Python 3.9-compatible typing (`Optional`, `List`, `Dict`, `Tuple`)
  unless the local file already consistently uses newer syntax
- Group imports as standard library, third-party, then local
- Use `CamelCase` for classes and `snake_case` for functions/methods
- Preserve established naming in ports or math-heavy code when cleanup would
  hurt correspondence with papers or upstream implementations

- Follow the existing test style in the directory you edit; do not rewrite test
  frameworks without a clear reason
- Add focused regression tests near the subsystem they cover
- Mark environment-sensitive tests carefully when they need solvers, network,
  or heavyweight tools
- In research/ported code, preserve behavior over stylistic cleanup
- Do not simplify algorithms unless you understand the proof/search impact
- Keep paper terminology and solver terminology aligned with existing code
- Document assumptions when changing heuristics, encodings, or search loops

- Do not mass-reformat `benchmarks/` or change benchmark semantics to satisfy a test
- Keep scripts runnable from repo root unless they already assume another cwd
- Preserve experiment-script CLI behavior unless the task explicitly changes it
- Check local guidance in `aria/efmc/AGENTS.md`, `aria/quant/AGENTS.md`,
  `aria/llmtools/AGENTS.md`, and `scripts/AGENTS.md`
