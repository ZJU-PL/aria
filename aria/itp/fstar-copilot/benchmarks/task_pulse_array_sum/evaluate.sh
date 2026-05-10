#!/usr/bin/env bash
# Evaluator for task_pulse_array_sum
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

FILE="$WORKSPACE/ArraySum.fst"

if [ ! -f "$FILE" ]; then
    record_score "$TASK_NAME" "verification" 0 40 "File not found"
    exit 0
fi

admit_score=10
if ! check_no_admits "$FILE"; then admit_score=0; fi
record_score "$TASK_NAME" "no-admits" "$admit_score" 10

verify_score=0
if verify_fst "$FILE" > "$RESULTS_DIR/${TASK_NAME}.fstar_output" 2>&1; then
    echo -e "${GREEN}Verification succeeded${NC}"
    verify_score=20
else
    echo -e "${RED}Verification failed${NC}"
    tail -20 "$RESULTS_DIR/${TASK_NAME}.fstar_output"
fi
record_score "$TASK_NAME" "verification" "$verify_score" 20

rlimit_score=0
if check_rlimit "$FILE" 10; then rlimit_score=5
elif check_rlimit "$FILE" 50; then rlimit_score=3; fi
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5

# Check key elements
elem_score=0
for pat in "sum_spec" "array_sum" "while" "invariant" "pure.*sum_spec"; do
    if grep -qE "$pat" "$FILE"; then
        elem_score=$((elem_score + 1))
    fi
done
record_score "$TASK_NAME" "key-elements" "$elem_score" 5
