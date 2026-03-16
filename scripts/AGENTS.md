# AGENTS.md - Scripts

`scripts/` contains evaluation utilities, SMT-COMP helpers, solver wrappers,
query collection tools, and one-off research scripts.

Read `scripts/README.md` and any local README in a subdirectory first.

## Working Rules

- Treat scripts as operator-facing tools, not polished library APIs.
- Preserve existing command-line behavior unless the task explicitly requests an
  interface change.
- Keep repo-root execution working when that is how the script is currently
  documented or implemented.
- Avoid broad cleanup changes in SMT-COMP or evaluation scripts unless you have
  verified the exact workflow they support.
- If a script depends on solver binaries, external datasets, or benchmark
  layouts, mention that dependency in your summary.

## Validation

- Prefer `--help` or a minimal smoke run when possible.
- For evaluation scripts, validate argument parsing and obvious import/runtime
  errors before making larger claims.
