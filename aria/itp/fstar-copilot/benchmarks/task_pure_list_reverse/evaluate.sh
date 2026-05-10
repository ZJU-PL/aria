#!/usr/bin/env bash
# Evaluator for task_pure_list_reverse
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

FILE="$WORKSPACE/ListReverse.fst"

# ── Check file exists ────────────────────────────────────────────────
if [ ! -f "$FILE" ]; then
    echo -e "${RED}ListReverse.fst not found in workspace${NC}"
    record_score "$TASK_NAME" "verification" 0 40 "File not found"
    exit 0
fi

# ── Deterministic: No admits ─────────────────────────────────────────
admit_score=10
if ! check_no_admits "$FILE"; then
    admit_score=0
fi
record_score "$TASK_NAME" "no-admits" "$admit_score" 10

# ── Deterministic: Verification ──────────────────────────────────────
verify_score=0
echo -e "${YELLOW}Verifying ListReverse.fst...${NC}"
if verify_fst "$FILE" > "$RESULTS_DIR/${TASK_NAME}.fstar_output" 2>&1; then
    echo -e "${GREEN}Verification succeeded${NC}"
    verify_score=20
else
    echo -e "${RED}Verification failed${NC}"
    tail -20 "$RESULTS_DIR/${TASK_NAME}.fstar_output"
fi
record_score "$TASK_NAME" "verification" "$verify_score" 20

# ── Deterministic: rlimit check ─────────────────────────────────────
rlimit_score=0
if check_rlimit "$FILE" 10; then
    rlimit_score=5
elif check_rlimit "$FILE" 50; then
    rlimit_score=3
fi
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5

# ── Deterministic: Required definitions present ──────────────────────
completeness_score=0
for fn in reverse_spec reverse reverse_correct reverse_involutive reverse_length reverse_mem; do
    if grep -qE "(val|let)\s+$fn\b" "$FILE"; then
        completeness_score=$((completeness_score + 1))
    else
        echo -e "${YELLOW}Missing: $fn${NC}"
    fi
done
# Scale: 6 functions -> max 5 points
completeness_score=$(( completeness_score * 5 / 6 ))
record_score "$TASK_NAME" "required-defs" "$completeness_score" 5
