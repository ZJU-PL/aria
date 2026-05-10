#!/usr/bin/env bash
# ── run_bench.sh ─────────────────────────────────────────────────────
#
# End-to-end benchmark scaffold:
#   1. Creates a fresh run directory
#   2. Builds F* from source (or reuses an existing build)
#   3. Launches `copilot` on each benchmark task in its own workspace
#   4. Runs deterministic evaluation and prepares LLM judge inputs
#
# Usage:
#   ./benchmarks/run_bench.sh                          # full run
#   ./benchmarks/run_bench.sh --skip-build             # reuse existing F* build
#   ./benchmarks/run_bench.sh --evaluate-only          # just evaluate
#   ./benchmarks/run_bench.sh task_pulse_swap           # single task
#   ./benchmarks/run_bench.sh --fstar-home /path/FStar  # use pre-built F*
#
# Environment:
#   COPILOT_BIN    Path to copilot binary   (default: copilot)
#   MODEL          Model to use             (default: claude-opus-4.6)
#   EFFORT         Reasoning effort level   (default: high)
#   FSTAR_HOME     Reuse existing F* build  (skips setup_fstar.sh)
#   JOBS           Build parallelism        (default: nproc)
#   TIMEOUT        Per-task timeout seconds (default: 600)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────
BENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
COPILOT_BIN="${COPILOT_BIN:-copilot}"
MODEL="${MODEL:-claude-opus-4.6}"
EFFORT="${EFFORT:-high}"
TIMEOUT="${TIMEOUT:-600}"
SKIP_BUILD=0
EVALUATE_ONLY=false
TASK_FILTER=""
FSTAR_HOME="${FSTAR_HOME:-}"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)      SKIP_BUILD=1; shift ;;
        --evaluate-only)   EVALUATE_ONLY=true; shift ;;
        --fstar-home)      FSTAR_HOME="$2"; SKIP_BUILD=1; shift 2 ;;
        --fstar-home=*)    FSTAR_HOME="${1#*=}"; SKIP_BUILD=1; shift ;;
        --model)           MODEL="$2"; shift 2 ;;
        --model=*)         MODEL="${1#*=}"; shift ;;
        --effort)          EFFORT="$2"; shift 2 ;;
        --effort=*)        EFFORT="${1#*=}"; shift ;;
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --timeout=*)       TIMEOUT="${1#*=}"; shift ;;
        --help|-h)
            sed -n '/^# ── run_bench/,/^# ──────/{s/^# \?//;p}' "$0" | head -20
            exit 0
            ;;
        task_*)            TASK_FILTER="$1"; shift ;;
        *)                 echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Create run directory ─────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$BENCH_ROOT/_bench_runs/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# Symlink latest
ln -sfn "run_${TIMESTAMP}" "$BENCH_ROOT/_bench_runs/latest"

echo -e "${BOLD}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   F* Copilot Benchmark — $TIMESTAMP   ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════╝${NC}"
echo ""
echo "  Run directory : $RUN_DIR"
echo "  Plugin source : $REPO_ROOT"
echo "  Model         : $MODEL"
echo "  Effort        : $EFFORT"
echo "  Timeout/task  : ${TIMEOUT}s"
echo ""

# ── Phase 1: Build F* from source ───────────────────────────────────
# Note: We use --config-dir with a clean temp directory when invoking
# copilot, so installed plugins are never loaded. No need to uninstall
# anything from the user's real config.
echo -e "${BLUE}── Phase 1: F* Toolchain ──${NC}"

# Shared F* build location: benchmarks/_fstar (persists across runs)
SHARED_FSTAR="$BENCH_ROOT/_fstar"

if [ -n "$FSTAR_HOME" ] && [ -x "$FSTAR_HOME/bin/fstar.exe" ]; then
    echo "  Using pre-built F* at $FSTAR_HOME"
    "$FSTAR_HOME/bin/fstar.exe" --version 2>&1 | head -1
elif [ -x "$SHARED_FSTAR/bin/fstar.exe" ] && [ "$SKIP_BUILD" = "1" ]; then
    FSTAR_HOME="$SHARED_FSTAR"
    echo "  Using shared F* build at $FSTAR_HOME (SKIP_BUILD=1)"
    "$FSTAR_HOME/bin/fstar.exe" --version 2>&1 | head -1
else
    FSTAR_HOME="$SHARED_FSTAR"
    echo "  Building F* in $FSTAR_HOME (shared across runs)..."
    echo "  (This will take a while — follow progress in $RUN_DIR/fstar_build.log)"
    SKIP_BUILD="$SKIP_BUILD" \
        "$BENCH_ROOT/setup_fstar.sh" "$FSTAR_HOME" \
        > "$RUN_DIR/fstar_build.log" 2>&1 \
    || {
        echo -e "  ${RED}F* build failed! See $RUN_DIR/fstar_build.log${NC}"
        exit 1
    }
    echo -e "  ${GREEN}F* build succeeded.${NC}"
fi

FSTAR_EXE="$FSTAR_HOME/bin/fstar.exe"
KRML_EXE="$FSTAR_HOME/karamel/krml"
echo "  fstar.exe: $FSTAR_EXE"
echo "  krml:      $KRML_EXE"
echo ""

# Record build info
{
    echo "fstar_home=$FSTAR_HOME"
    echo "fstar_exe=$FSTAR_EXE"
    echo "krml_exe=$KRML_EXE"
    echo "model=$MODEL"
    echo "effort=$EFFORT"
    echo "timeout=$TIMEOUT"
    echo "timestamp=$TIMESTAMP"
    echo "plugin_dir=$REPO_ROOT"
    "$FSTAR_EXE" --version 2>&1 | head -1 | sed 's/^/fstar_version=/'
} > "$RUN_DIR/run_info.txt"

# ── Phase 2: Discover tasks ─────────────────────────────────────────
echo -e "${BLUE}── Phase 2: Discovering tasks ──${NC}"
TASKS=()
if [ -n "$TASK_FILTER" ]; then
    if [ -d "$BENCH_ROOT/$TASK_FILTER" ]; then
        TASKS+=("$TASK_FILTER")
    else
        echo -e "${RED}Task not found: $TASK_FILTER${NC}" >&2
        exit 1
    fi
else
    for d in "$BENCH_ROOT"/task_*/; do
        [ -d "$d" ] && TASKS+=("$(basename "$d")")
    done
fi

echo "  Tasks: ${#TASKS[@]}"
for t in "${TASKS[@]}"; do echo "    • $t"; done
echo ""

# ── Phase 3: Run agent on each task ─────────────────────────────────
echo -e "${BLUE}── Phase 3: Running agent on tasks ──${NC}"
RESULTS_DIR="$RUN_DIR/results"
mkdir -p "$RESULTS_DIR"

# Use a temporary, clean copilot config directory.
# We copy auth credentials from the user's real config but register ONLY the
# local plugin (from $REPO_ROOT), so the agent uses the source tree rather than
# any globally-installed version.
CLEAN_CONFIG_DIR="$RUN_DIR/.copilot-config"
mkdir -p "$CLEAN_CONFIG_DIR/installed-plugins/_direct"
ln -sfn "$REPO_ROOT" "$CLEAN_CONFIG_DIR/installed-plugins/_direct/local--fstar-copilot"

REAL_CONFIG="${HOME}/.copilot/config.json"
if [ -f "$REAL_CONFIG" ]; then
    python3 -c "
import json, sys
with open('$REAL_CONFIG') as f:
    cfg = json.load(f)
cfg['installed_plugins'] = [{
    'name': 'fstar-copilot',
    'marketplace': '',
    'version': '0.0.3',
    'installed_at': '2026-01-01T00:00:00.000Z',
    'enabled': True,
    'cache_path': '$CLEAN_CONFIG_DIR/installed-plugins/_direct/local--fstar-copilot',
    'source': {'source': 'local', 'path': '$REPO_ROOT'}
}]
with open('$CLEAN_CONFIG_DIR/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
" || {
        echo -e "  ${RED}Failed to create clean config.json${NC}" >&2
        exit 1
    }
    echo "  Clean config created at $CLEAN_CONFIG_DIR (auth from $REAL_CONFIG)"
else
    echo -e "  ${RED}No $REAL_CONFIG found — copilot may fail to authenticate${NC}" >&2
fi

run_agent_on_task() {
    local task="$1"
    local task_src="$BENCH_ROOT/$task"
    local task_run="$RUN_DIR/tasks/$task"
    local workspace="$task_run/workspace"
    local log_file="$RESULTS_DIR/${task}.agent_log"

    mkdir -p "$workspace"

    # Copy input files if any
    if [ -d "$task_src/input" ]; then
        cp -r "$task_src/input/"* "$workspace/"
    fi

    # Read the task prompt
    local prompt
    prompt="$(cat "$task_src/task.md")"

    # Prepend context about the F* build location
    local full_prompt
    full_prompt="$(cat <<PROMPT_EOF
You are working in: $workspace

The F* toolchain is available at:
  FSTAR_HOME=$FSTAR_HOME
  fstar.exe=$FSTAR_EXE
  krml=$KRML_EXE

All output files should be placed in the current working directory ($workspace).

--- TASK ---

$prompt
PROMPT_EOF
)"

    echo -e "  ${YELLOW}[$task]${NC} Starting agent..."

    # Launch copilot with:
    #   --config-dir        → clean dir with only the local plugin registered
    #   --agent             → the fstar-coder agent from the local plugin
    #   --no-custom-instructions → ignore AGENTS.md etc. from workspace
    #   -p                  → non-interactive prompt mode
    #   --allow-all-tools   → no confirmation prompts
    #   --allow-all-paths   → allow file access everywhere
    #   --model             → specific model
    #   --effort            → reasoning effort
    timeout "$TIMEOUT" \
        "$COPILOT_BIN" \
            --config-dir "$CLEAN_CONFIG_DIR" \
            --agent "fstar-copilot:fstar-coder" \
            --no-custom-instructions \
            --model "$MODEL" \
            --effort "$EFFORT" \
            --allow-all-tools \
            --allow-all-paths \
            --no-color \
            --add-dir "$workspace" \
            --add-dir "$FSTAR_HOME" \
            -p "$full_prompt" \
        > "$log_file" 2>&1 \
    && status=0 || status=$?

    if [ $status -eq 0 ]; then
        echo -e "  ${GREEN}[$task]${NC} Agent completed successfully."
    elif [ $status -eq 124 ]; then
        echo -e "  ${RED}[$task]${NC} Agent timed out after ${TIMEOUT}s."
    else
        echo -e "  ${RED}[$task]${NC} Agent exited with status $status."
    fi

    echo "$status" > "$RESULTS_DIR/${task}.exit_status"
}

if [ "$EVALUATE_ONLY" = true ]; then
    echo "  Skipping agent execution (--evaluate-only)"
    # Point workspace paths to existing run dir
    for task in "${TASKS[@]}"; do
        task_run="$RUN_DIR/tasks/$task"
        if [ ! -d "$task_run/workspace" ]; then
            echo -e "  ${RED}No workspace for $task — skipping${NC}"
        fi
    done
else
    for task in "${TASKS[@]}"; do
        run_agent_on_task "$task"
    done
fi
echo ""

# ── Phase 4: Evaluate ───────────────────────────────────────────────
echo -e "${BLUE}── Phase 4: Evaluation ──${NC}"

source "$BENCH_ROOT/lib/common.sh"
export FSTAR_EXE KRML_EXE RESULTS_DIR

for task in "${TASKS[@]}"; do
    task_src="$BENCH_ROOT/$task"
    task_run="$RUN_DIR/tasks/$task"
    workspace="$task_run/workspace"
    results_file="$RESULTS_DIR/${task}.scores"

    echo -e "  ${YELLOW}[$task]${NC} Evaluating..."

    # Clear previous scores
    rm -f "$results_file"

    if [ ! -d "$workspace" ]; then
        echo -e "  ${RED}[$task]${NC} No workspace — skipping"
        continue
    fi

    # Count files produced
    file_count=$(find "$workspace" \( -name '*.fst' -o -name '*.fsti' -o -name 'Makefile' \) | wc -l)
    echo -e "  ${YELLOW}[$task]${NC} Files produced: $file_count"

    # Run task-specific evaluator
    if [ -x "$task_src/evaluate.sh" ]; then
        (
            export TASK_NAME="$task"
            export TASK_DIR="$task_src"
            export WORKSPACE="$workspace"
            "$task_src/evaluate.sh"
        ) || echo -e "  ${RED}[$task]${NC} Evaluator had errors"
    fi

    # Prepare LLM judge input
    judge_input="$RESULTS_DIR/${task}.judge_input"
    prepare_judge_prompt "$task" "$task_src/task.md" "$workspace" "$judge_input"

    print_task_summary "$task"
done

# ── Phase 5: Summary ────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║             BENCHMARK SUMMARY              ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════╝${NC}"
echo ""
printf "  %-35s %10s  %s\n" "Task" "Score" "Agent Exit"
printf "  %-35s %10s  %s\n" "───────────────────────────────────" "──────────" "──────────"

grand_total=0
grand_max=0

for task in "${TASKS[@]}"; do
    t=$(total_score "$task")
    m=$(max_score "$task")
    grand_total=$((grand_total + t))
    grand_max=$((grand_max + m))

    exit_file="$RESULTS_DIR/${task}.exit_status"
    exit_status="—"
    if [ -f "$exit_file" ]; then
        exit_status="$(cat "$exit_file")"
        case "$exit_status" in
            0)   exit_status="${GREEN}OK${NC}" ;;
            124) exit_status="${RED}TIMEOUT${NC}" ;;
            *)   exit_status="${RED}EXIT $exit_status${NC}" ;;
        esac
    fi

    if (( m > 0 )); then
        pct=$((t * 100 / m))
    else
        pct=0
    fi
    printf "  %-35s %4d/%-4d  " "$task" "$t" "$m"
    echo -e "$exit_status"
done

echo ""
if (( grand_max > 0 )); then
    grand_pct=$((grand_total * 100 / grand_max))
else
    grand_pct=0
fi
echo -e "  ${BOLD}GRAND TOTAL: $grand_total / $grand_max ($grand_pct%)${NC}"
echo ""
echo "  Run directory : $RUN_DIR"
echo "  Results       : $RESULTS_DIR/"
echo "  Judge inputs  : $RESULTS_DIR/*.judge_input"
echo ""
echo "  To re-evaluate:  $0 --evaluate-only --fstar-home $FSTAR_HOME"
echo ""

# Write machine-readable summary
{
    echo "timestamp=$TIMESTAMP"
    echo "grand_total=$grand_total"
    echo "grand_max=$grand_max"
    echo "grand_pct=$grand_pct"
    echo "tasks=${#TASKS[@]}"
    echo "model=$MODEL"
    echo "effort=$EFFORT"
} > "$RESULTS_DIR/summary.txt"
