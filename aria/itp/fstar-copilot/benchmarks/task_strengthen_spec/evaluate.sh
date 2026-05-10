#!/usr/bin/env bash
# Evaluator for task_strengthen_spec
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

FSTI="$WORKSPACE/WeakSpec.fsti"
FST="$WORKSPACE/WeakSpec.fst"

# File existence (2 points)
files=0
[ -f "$FSTI" ] && files=$((files + 1))
[ -f "$FST" ] && files=$((files + 1))
record_score "$TASK_NAME" "files-exist" "$files" 2

# No admits (8 points)
admit_score=8
for f in "$FSTI" "$FST"; do
    if [ -f "$f" ] && ! check_no_admits "$f"; then
        admit_score=0; break
    fi
done
# Check any optional StoreSpec.fst too
if [ -f "$WORKSPACE/StoreSpec.fst" ] && ! check_no_admits "$WORKSPACE/StoreSpec.fst"; then
    admit_score=0
fi
record_score "$TASK_NAME" "no-admits" "$admit_score" 8

# Verification (15 points)
verify_score=0
all_ok=true
for f in "$FSTI" "$FST"; do
    if [ -f "$f" ]; then
        if verify_fst "$f" --include "$WORKSPACE" > "$RESULTS_DIR/${TASK_NAME}.$(basename "$f").out" 2>&1; then
            echo -e "${GREEN}  $(basename "$f") OK${NC}"
        else
            echo -e "${RED}  $(basename "$f") FAILED${NC}"
            all_ok=false
        fi
    fi
done
if [ -f "$WORKSPACE/StoreSpec.fst" ]; then
    if verify_fst "$WORKSPACE/StoreSpec.fst" --include "$WORKSPACE" \
        > "$RESULTS_DIR/${TASK_NAME}.StoreSpec.fst.out" 2>&1; then
        echo -e "${GREEN}  StoreSpec.fst OK${NC}"
    else
        echo -e "${RED}  StoreSpec.fst FAILED${NC}"
        all_ok=false
    fi
fi
$all_ok && verify_score=15
record_score "$TASK_NAME" "verification" "$verify_score" 15

# Spec strength heuristic: check for spec-model references in postconditions (10 points)
strength=0
if [ -f "$FSTI" ]; then
    # Look for return value constraints in postconditions
    grep -qE 'fun\s+\w+\s*->' "$FSTI" && strength=$((strength + 2))
    # Look for spec model references
    grep -qiE 'spec\|model\|logical\|Map\|map' "$FSTI" && strength=$((strength + 3))
    # Look for meaningful postconditions (not just exists* with no pure)
    grep -qE 'pure.*==' "$FSTI" && strength=$((strength + 3))
    # Look for return value binding in ensures
    grep -qE 'ensures.*fun.*r\|returns.*ensures.*pure.*r' "$FSTI" && strength=$((strength + 2))
fi
record_score "$TASK_NAME" "spec-strength" "$strength" 10

# rlimit
max_rl=0
for f in "$FST" "$WORKSPACE/StoreSpec.fst"; do
    [ -f "$f" ] && { m=$(max_rlimit "$f"); (( m > max_rl )) && max_rl=$m; }
done
rlimit_score=0
(( max_rl <= 10 )) && rlimit_score=5 || { (( max_rl <= 50 )) && rlimit_score=3; }
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5
