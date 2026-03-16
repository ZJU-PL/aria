# AGENTS.md - Quantified Reasoning

`aria.quant` collects quantified reasoning code, including EFSMT, CHC tooling,
quantifier elimination experiments, and multiple research prototypes.

Read `aria/quant/README.md` before making broad changes here.

## Important Context

- This package is explicitly heterogeneous in maturity.
- Several subpackages are self-contained research artifacts.
- External solver binaries may be required for some tactics or experiments.
- Similar concepts may appear in more than one implementation; do not assume
  there is a single canonical abstraction.

## Key Areas

- `efsmt_parser.py`, `efsmt_solver.py`, `efsmt_utils.py`: shared front-end glue
- `efbool/`, `efbv/`, `eflira/`: exists-forall stacks for different theories
- `qe/`: quantifier elimination experiments and adapters
- `chctools/`, `polyhorn/`, `fossil/`, `ufbv/`: specialized solver/prototype code

## Working Rules

- Before refactoring, identify whether a file is shared infrastructure or a
  theory-specific experiment.
- Preserve theory-specific behavior and terminology such as BV, LIA, LRA, CHC,
  CEGAR, QBF reduction, and sampling-based solving.
- Do not normalize all solver interfaces unless the task explicitly asks for it;
  inconsistency here is often historical and intentional.
- If a tactic shells out to an external solver, keep fallback behavior and error
  reporting explicit.

## Testing

Prefer narrow tests near the changed implementation:

```bash
pytest aria/quant/efbv/tests
pytest aria/quant/eflira/tests
pytest aria/tests/test_efbv.py
pytest aria/tests/test_qe.py
```

If a test depends on optional binaries or sampling, mention that in your change
summary and avoid claiming deterministic coverage you did not get.
