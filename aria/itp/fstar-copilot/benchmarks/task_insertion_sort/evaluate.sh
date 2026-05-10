#!/usr/bin/env bash
# Evaluator for task_insertion_sort
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

SPEC="$WORKSPACE/InsertionSort.Spec.fst"
IMPL="$WORKSPACE/InsertionSort.Impl.fst"
INTF="$WORKSPACE/InsertionSort.Impl.fsti"

# File existence
files_found=0
for f in "$SPEC" "$IMPL" "$INTF"; do
    if [ -f "$f" ]; then
        files_found=$((files_found + 1))
    else
        echo -e "${RED}Missing: $(basename "$f")${NC}"
    fi
done
record_score "$TASK_NAME" "file-structure" $(( files_found * 3 / 3 )) 3

# No admits across all files
admit_score=7
for f in "$SPEC" "$IMPL" "$INTF"; do
    if [ -f "$f" ] && ! check_no_admits "$f"; then
        admit_score=0
        break
    fi
done
record_score "$TASK_NAME" "no-admits" "$admit_score" 7

# Verification
verify_score=0
all_ok=true
for f in "$INTF" "$SPEC" "$IMPL"; do
    if [ -f "$f" ]; then
        echo -e "${YELLOW}Verifying $(basename "$f")...${NC}"
        if verify_fst "$f" --include "$WORKSPACE" > "$RESULTS_DIR/${TASK_NAME}.$(basename "$f").out" 2>&1; then
            echo -e "${GREEN}  $(basename "$f") OK${NC}"
        else
            echo -e "${RED}  $(basename "$f") FAILED${NC}"
            tail -10 "$RESULTS_DIR/${TASK_NAME}.$(basename "$f").out"
            all_ok=false
        fi
    fi
done
if $all_ok && [ $files_found -eq 3 ]; then
    verify_score=20
elif $all_ok; then
    verify_score=10
fi
record_score "$TASK_NAME" "verification" "$verify_score" 20

# rlimit
max_rl=0
for f in "$SPEC" "$IMPL"; do
    if [ -f "$f" ]; then
        m=$(max_rlimit "$f")
        if (( m > max_rl )); then max_rl=$m; fi
    fi
done
rlimit_score=0
if (( max_rl <= 10 )); then rlimit_score=5
elif (( max_rl <= 20 )); then rlimit_score=3
elif (( max_rl <= 50 )); then rlimit_score=1; fi
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5

# Key spec properties
spec_score=0
if [ -f "$IMPL" ]; then
    grep -qE "sorted" "$IMPL" && spec_score=$((spec_score + 2))
    grep -qE "permutation\|perm" "$IMPL" && spec_score=$((spec_score + 3))
fi
record_score "$TASK_NAME" "spec-properties" "$spec_score" 5
