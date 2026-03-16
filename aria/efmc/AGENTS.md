# AGENTS.md - EFMC

`aria.efmc` is the verification-oriented part of the repo. It includes
frontends, transition-system construction, multiple proving engines, template
families, and CLI entrypoints.

## Start Here

Useful files for orientation:

- `aria/efmc/cli/efmc.py`: main verification CLI
- `aria/efmc/cli/efsmt.py`: EFSMT-oriented CLI entrypoint in this subsystem
- `aria/efmc/frontends/`: parsers for CHC, SyGuS, Boogie, and C inputs
- `aria/efmc/engines/`: proving engines such as EF, PDR, k-induction, abduction
- `aria/efmc/tests/`: main regression suite

## Working Rules

- Understand which input format is in play before editing parser or CLI code.
- Many code paths branch on theory: bit-vectors, integers/reals, or floating
  point. Preserve those distinctions.
- Template selection is central to EF-based proving. If you change template
  logic, test both selection and solver invocation paths.
- Be careful with global configuration objects such as verifier args; avoid
  introducing hidden cross-test state.
- Some engines lazily import less-common provers. Preserve that pattern unless
  you are intentionally changing startup/import cost.

## Testing

Prefer targeted tests:

```bash
pytest aria/efmc/tests/test_cli_efmc.py
pytest aria/efmc/tests/test_cli_efsmt.py
pytest aria/efmc/tests/test_kinduction.py
pytest aria/efmc/tests/test_termination.py
pytest aria/efmc/tests/test_boogie_converter.py
```

For CLI changes, also run:

```bash
python -m aria.efmc.cli.efmc --help
python -m aria.efmc.cli.efsmt --help
python -m aria.efmc.cli.polyhorn --help
```

## Common Risks

- regressions in signed/unsigned bit-vector handling
- template selection mismatches for BV vs FP vs arithmetic systems
- parser/CLI drift across CHC, SyGuS, Boogie, and C frontends
- timeouts or flaky tests caused by solver search changes
