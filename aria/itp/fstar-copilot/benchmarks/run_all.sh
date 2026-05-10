#!/usr/bin/env bash
# ── F* Copilot Benchmark Runner ──────────────────────────────────────
#
# Usage:
#   ./benchmarks/run_all.sh                       # Run all tasks
#   ./benchmarks/run_all.sh task_pure_list_reverse # Run one task
#   ./benchmarks/run_all.sh --evaluate-only        # Evaluate existing outputs
#   ./benchmarks/run_all.sh --list                 # List available tasks
#
# Environment:
#   FSTAR_HOME   Path to built fstar2 checkout (required for verification)
#   FSTAR_EXE    Override path to fstar.exe
#   AGENT_CMD    Command to invoke the agent (default: copilot -a fstar-copilot:fstar-coder)
#   RESULTS_DIR  Where to write scores (default: benchmarks/_results)
#
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$BENCH_ROOT/lib/common.sh"

AGENT_CMD="${AGENT_CMD:-copilot -a fstar-copilot:fstar-coder}"
EVALUATE_ONLY=false
TASK_FILTER=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --evaluate-only) EVALUATE_ONLY=true; shift ;;
        --list)
            echo "Available benchmark tasks:"
            for d in "$BENCH_ROOT"/task_*/; do
                name="$(basename "$d")"
                desc=""
                if [ -f "$d/task.md" ]; then
                    desc="$(head -1 "$d/task.md" | sed 's/^#* *//')"
                fi
                printf "  %-35s %s\n" "$name" "$desc"
            done
            exit 0
            ;;
        --help|-h)
            head -12 "$0" | tail -11 | sed 's/^# *//'
            exit 0
            ;;
        task_*)  TASK_FILTER="$1"; shift ;;
        *)       echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Discover tasks ───────────────────────────────────────────────────
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
        TASKS+=("$(basename "$d")")
    done
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    echo -e "${RED}No benchmark tasks found.${NC}" >&2
    exit 1
fi

echo -e "${BOLD}F* Copilot Benchmark Suite${NC}"
echo -e "Tasks to run: ${#TASKS[@]}"
echo -e "FSTAR_HOME:   ${FSTAR_HOME:-<not set>}"
echo -e "Results dir:  $RESULTS_DIR"
echo

# ── Run each task ────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

for task in "${TASKS[@]}"; do
    task_dir="$BENCH_ROOT/$task"
    workspace="$task_dir/workspace"
    task_md="$task_dir/task.md"
    results_file="$RESULTS_DIR/${task}.scores"

    echo -e "${BLUE}━━━ $task ━━━${NC}"

    # Clear previous scores
    rm -f "$results_file"

    # ── Phase 1: Agent execution ─────────────────────────────────
    if [ "$EVALUATE_ONLY" = false ]; then
        # Prepare workspace
        rm -rf "$workspace"
        mkdir -p "$workspace"

        # Copy input files if any
        if [ -d "$task_dir/input" ]; then
            cp -r "$task_dir/input/"* "$workspace/" 2>/dev/null || true
        fi

        echo -e "  ${YELLOW}Running agent...${NC}"
        prompt="$(cat "$task_md")"

        # The agent works in the workspace directory.
        # The actual invocation depends on the agent framework.
        # This produces a transcript that can be reviewed.
        (
            cd "$workspace"
            echo "$prompt" | $AGENT_CMD \
                --prompt-stdin \
                2>&1 | tee "$RESULTS_DIR/${task}.agent_log"
        ) || echo -e "  ${YELLOW}Agent exited with non-zero status${NC}"
    fi

    # ── Phase 2: Evaluation ──────────────────────────────────────
    if [ ! -d "$workspace" ]; then
        echo -e "  ${RED}No workspace found; skipping evaluation${NC}"
        continue
    fi

    echo -e "  ${YELLOW}Evaluating...${NC}"

    # Run the task-specific evaluator
    if [ -x "$task_dir/evaluate.sh" ]; then
        (
            export TASK_NAME="$task"
            export TASK_DIR="$task_dir"
            export WORKSPACE="$workspace"
            export RESULTS_DIR
            export FSTAR_EXE
            source "$BENCH_ROOT/lib/common.sh"
            "$task_dir/evaluate.sh"
        ) || echo -e "  ${RED}Evaluator failed${NC}"
    else
        echo -e "  ${YELLOW}No evaluator found${NC}"
    fi

    # ── Phase 3: LLM judge ───────────────────────────────────────
    judge_input="$RESULTS_DIR/${task}.judge_input"
    prepare_judge_prompt "$task" "$task_md" "$workspace" "$judge_input"
    echo -e "  ${YELLOW}Judge prompt written to: $judge_input${NC}"
    echo -e "  ${YELLOW}(Run your LLM judge on this file to get Correctness/Style/Completeness scores)${NC}"

    print_task_summary "$task"
done

# ── Final summary ────────────────────────────────────────────────────
echo -e "${BOLD}═══ SUMMARY ═══${NC}"
printf "%-35s %8s\n" "Task" "Score"
printf "%-35s %8s\n" "───────────────────────────────────" "────────"
grand_total=0
grand_max=0
for task in "${TASKS[@]}"; do
    t=$(total_score "$task")
    m=$(max_score "$task")
    grand_total=$((grand_total + t))
    grand_max=$((grand_max + m))
    if (( m > 0 )); then
        pct=$((t * 100 / m))
    else
        pct=0
    fi
    printf "%-35s %4d/%-4d (%d%%)\n" "$task" "$t" "$m" "$pct"
done
echo -e "${BOLD}"
printf "%-35s %4d/%-4d" "GRAND TOTAL" "$grand_total" "$grand_max"
if (( grand_max > 0 )); then
    printf " (%d%%)" $((grand_total * 100 / grand_max))
fi
printf "\n"
echo -e "${NC}"
