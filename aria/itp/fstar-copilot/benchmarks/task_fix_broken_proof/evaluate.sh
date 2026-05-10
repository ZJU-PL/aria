#!/usr/bin/env bash
# Evaluator for task_fix_broken_proof
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

FILE="$WORKSPACE/BrokenLemmas.fst"

if [ ! -f "$FILE" ]; then
    record_score "$TASK_NAME" "verification" 0 40 "File not found"
    exit 0
fi

# No admits (10 points)
admit_score=10
if ! check_no_admits "$FILE"; then admit_score=0; fi
record_score "$TASK_NAME" "no-admits" "$admit_score" 10

# Verification (20 points)
verify_score=0
if verify_fst "$FILE" > "$RESULTS_DIR/${TASK_NAME}.fstar_output" 2>&1; then
    echo -e "${GREEN}Verification succeeded${NC}"
    verify_score=20
else
    echo -e "${RED}Verification failed${NC}"
    tail -20 "$RESULTS_DIR/${TASK_NAME}.fstar_output"
fi
record_score "$TASK_NAME" "verification" "$verify_score" 20

# Lemma statements preserved (5 points)
# Check that the original val declarations are still present
preserved=0
for lemma in "append_length" "rev_append" "map_compose" "mem_append" "length_filter_le"; do
    if grep -qE "val\s+$lemma" "$FILE"; then
        preserved=$((preserved + 1))
    else
        echo -e "${YELLOW}Missing val for: $lemma${NC}"
    fi
done
record_score "$TASK_NAME" "lemmas-preserved" "$preserved" 5

# rlimit
rlimit_score=0
if check_rlimit "$FILE" 10; then rlimit_score=5
elif check_rlimit "$FILE" 50; then rlimit_score=3; fi
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5
