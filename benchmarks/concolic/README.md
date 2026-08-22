# Native concolic benchmark suite

`manifest.json` contains reproducible function-level campaigns for an Aria
multi-module fixture and source-available Python standard-library code. Add real
library targets by specifying an import target, seeds, packages to instrument,
coverage sources, and exploration budget.

Run from the repository root:

```bash
python scripts/run_concolic_benchmarks.py \
  --manifest benchmarks/concolic/manifest.json \
  --output-json /tmp/concolic-benchmarks.json \
  --output-csv /tmp/concolic-benchmarks.csv
```

Each case runs a one-seed baseline and a full concolic campaign in fresh
subprocesses. Reports compare executions, paths, failures, elapsed time,
frontier queries, and conventional line/function/branch coverage. Targets may
be filtered with `--filter SUBSTRING`.

