#!/usr/bin/env bash
# Evaluator for task_binary_search_pulse
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

FILE="$WORKSPACE/BinarySearch.fst"

if [ ! -f "$FILE" ]; then
    echo -e "${RED}BinarySearch.fst not found${NC}"
    record_score "$TASK_NAME" "verification" 0 40 "File not found"
    exit 0
fi

# No admits
admit_score=10
if ! check_no_admits "$FILE"; then admit_score=0; fi
record_score "$TASK_NAME" "no-admits" "$admit_score" 10

# Verification
verify_score=0
if verify_fst "$FILE" > "$RESULTS_DIR/${TASK_NAME}.fstar_output" 2>&1; then
    echo -e "${GREEN}Verification succeeded${NC}"
    verify_score=20
else
    echo -e "${RED}Verification failed${NC}"
    tail -20 "$RESULTS_DIR/${TASK_NAME}.fstar_output"
fi
record_score "$TASK_NAME" "verification" "$verify_score" 20

# rlimit
rlimit_score=0
if check_rlimit "$FILE" 10; then rlimit_score=5
elif check_rlimit "$FILE" 50; then rlimit_score=3; fi
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5

# Key elements present
elem_score=0
for pat in "sorted" "binary_search" "loop" "invariant" "#lang-pulse\|Pulse.Lib"; do
    if grep -qE "$pat" "$FILE"; then
        elem_score=$((elem_score + 1))
    fi
done
elem_score=$(( elem_score * 5 / 4 ))
record_score "$TASK_NAME" "key-elements" "$elem_score" 5
