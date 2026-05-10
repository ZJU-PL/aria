# F* Copilot Benchmark Suite

Evaluation benchmarks for the `fstar-coder` agent and associated skills
(`fstarverifier`, `krmlextraction`, `projectsetup`, `proofdebugging`,
`smtprofiling`, `specreview`).

## Task Categories

| # | Task | Skills Exercised | Difficulty |
|---|------|-----------------|------------|
| 1 | `task_pure_list_reverse` | fstar-coder | Easy |
| 2 | `task_binary_search_pulse` | fstar-coder, fstarverifier | Medium |
| 3 | `task_pulse_array_sum` | fstar-coder, fstarverifier | Medium |
| 4 | `task_pulse_swap` | fstar-coder | Easy |
| 5 | `task_insertion_sort` | fstar-coder, specreview | Hard |
| 6 | `task_counter_extraction` | fstar-coder, krmlextraction, projectsetup | Hard |
| 7 | `task_fix_broken_proof` | proofdebugging, fstarverifier | Medium |
| 8 | `task_fix_pulse_errors` | proofdebugging, fstarverifier | Medium |
| 9 | `task_strengthen_spec` | specreview, fstar-coder | Medium |
| 10 | `task_project_setup` | projectsetup | Easy |
| 11 | `task_analyze_and_improve` | specreview, proofdebugging, fstar-coder | Hard |

## Running

```bash
# Full end-to-end run: build F*, run agent on all tasks, evaluate
./benchmarks/run_bench.sh

# Use a pre-built F*
./benchmarks/run_bench.sh --fstar-home /path/to/FStar

# Run a single task
./benchmarks/run_bench.sh task_pure_list_reverse

# Skip F* build (reuse from previous run)
./benchmarks/run_bench.sh --skip-build

# Re-evaluate existing outputs
./benchmarks/run_bench.sh --evaluate-only --fstar-home /path/to/FStar

# Use a specific model
./benchmarks/run_bench.sh --model claude-opus-4-20250514 --effort xhigh
```

### What `run_bench.sh` does

1. Creates a **clean copilot config directory** with auth copied from
   `~/.copilot/config.json` but only the local plugin registered (symlinked
   from this repo). No globally-installed plugins are loaded.
2. **Builds F*** from the `fstar2` branch (or reuses an existing build).
3. For each task, **launches `copilot`** in a clean subdirectory with:
   - `--config-dir` pointing to a temporary empty config (no installed plugins)
   - `--plugin-dir` pointing to this repo (loads `agents/` + `skills/` from source)
   - `--agent fstar-copilot:fstar-coder`
4. **Evaluates** each task's output (deterministic checks + LLM judge prep).
5. **Reports** scores.

Each run is saved under `benchmarks/_bench_runs/run_<timestamp>/`.

## Evaluation

Each task is scored on a 100-point scale:

| Criterion | Points | Method |
|-----------|--------|--------|
| **Verification** | 0–40 | Deterministic: `fstar.exe` exit code, no admits/assumes |
| **Correctness** | 0–30 | LLM judge: specs match mathematical intent |
| **Style** | 0–15 | Mixed: rlimit ≤ 10 (deterministic) + code quality (LLM judge) |
| **Completeness** | 0–15 | LLM judge: all requirements addressed |

The deterministic checks run first. If verification fails (score 0 on
Verification), the remaining criteria are evaluated on the un-verified code
with a penalty.

## Directory Layout

```
benchmarks/
├── run_bench.sh            # End-to-end orchestrator (build + agent + evaluate)
├── setup_fstar.sh          # F* toolchain build script
├── run_all.sh              # Lightweight evaluator (agent invocation placeholder)
├── lib/
│   ├── common.sh           # Shared shell helpers
│   └── judge_prompt.md     # LLM evaluator system prompt
├── task_<name>/
│   ├── task.md             # Prompt sent to the agent
│   ├── evaluate.sh         # Task-specific evaluation script
│   ├── input/              # (optional) Files given to the agent
│   ├── reference/          # (optional) Reference solution / gold tests
│   └── workspace/          # Agent writes output here (gitignored)
├── _bench_runs/            # Run artifacts (gitignored)
│   └── run_<timestamp>/
│       ├── tools/FStar/    # F* build (if built by scaffold)
│       ├── tasks/<name>/workspace/  # Per-task agent workspace
│       ├── results/        # Scores, logs, judge inputs
│       └── run_info.txt    # Build + config metadata
```
