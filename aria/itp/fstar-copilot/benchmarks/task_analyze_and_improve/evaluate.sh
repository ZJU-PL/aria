#!/usr/bin/env bash
# Evaluator for task_analyze_and_improve
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

# Module split (5 points)
split_score=0
spec_file=$(find "$WORKSPACE" -name '*Spec*' -name '*.fst' 2>/dev/null | head -1)
impl_file=$(find "$WORKSPACE" -name '*Impl*' -name '*.fst' 2>/dev/null | head -1)
intf_file=$(find "$WORKSPACE" -name '*Impl*' -name '*.fsti' 2>/dev/null | head -1)
[ -n "$spec_file" ] && split_score=$((split_score + 2))
[ -n "$impl_file" ] && split_score=$((split_score + 2))
[ -n "$intf_file" ] && split_score=$((split_score + 1))
record_score "$TASK_NAME" "module-split" "$split_score" 5

# No admits across all files (10 points)
admit_score=10
for f in "$WORKSPACE"/*.fst "$WORKSPACE"/**/*.fst; do
    if [ -f "$f" ] && ! check_no_admits "$f" 2>/dev/null; then
        admit_score=0; break
    fi
done
record_score "$TASK_NAME" "no-admits" "$admit_score" 10

# Verification (15 points)
verify_score=0
all_ok=true
includes=""
for d in "$WORKSPACE" "$WORKSPACE"/spec "$WORKSPACE"/impl; do
    [ -d "$d" ] && includes="$includes --include $d"
done
for f in "$WORKSPACE"/*.fsti "$WORKSPACE"/**/*.fsti "$WORKSPACE"/*.fst "$WORKSPACE"/**/*.fst; do
    if [ -f "$f" ]; then
        echo -e "${YELLOW}Verifying $(basename "$f")...${NC}"
        if eval verify_fst "\"$f\"" $includes > "$RESULTS_DIR/${TASK_NAME}.$(basename "$f").out" 2>&1; then
            echo -e "${GREEN}  OK${NC}"
        else
            echo -e "${RED}  FAILED${NC}"
            all_ok=false
        fi
    fi
done
$all_ok && verify_score=15
record_score "$TASK_NAME" "verification" "$verify_score" 15

# rlimit (5 points)
max_rl=0
for f in "$WORKSPACE"/*.fst "$WORKSPACE"/**/*.fst; do
    if [ -f "$f" ]; then
        m=$(max_rlimit "$f")
        (( m > max_rl )) && max_rl=$m
    fi
done
rlimit_score=0
(( max_rl <= 10 )) && rlimit_score=5
(( max_rl > 10 && max_rl <= 20 )) && rlimit_score=4
(( max_rl > 20 && max_rl <= 50 )) && rlimit_score=2
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5

# Extraction readiness (5 points)
extract_score=0
if [ -n "$impl_file" ] && [ -f "$impl_file" ]; then
    grep -qE 'UInt64|UInt32|SizeT' "$impl_file" && extract_score=$((extract_score + 2))
    grep -qE 'inline_for_extraction' "$impl_file" && extract_score=$((extract_score + 2))
    grep -qE 'Ghost\.erased\|erased' "$impl_file" && extract_score=$((extract_score + 1))
fi
record_score "$TASK_NAME" "extraction-readiness" "$extract_score" 5
