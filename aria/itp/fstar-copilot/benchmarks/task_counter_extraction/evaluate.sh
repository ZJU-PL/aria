#!/usr/bin/env bash
# Evaluator for task_counter_extraction
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

SPEC="$WORKSPACE/spec/Counter.Spec.fst"
IMPL="$WORKSPACE/impl/Counter.Impl.fst"
INTF="$WORKSPACE/impl/Counter.Impl.fsti"
MKFILE="$WORKSPACE/Makefile"

# File structure (3 points)
files_found=0
for f in "$SPEC" "$IMPL" "$INTF" "$MKFILE"; do
    [ -f "$f" ] && files_found=$((files_found + 1))
done
record_score "$TASK_NAME" "file-structure" $(( files_found * 3 / 4 )) 3

# No admits (7 points)
admit_score=7
for f in "$SPEC" "$IMPL" "$INTF"; do
    if [ -f "$f" ] && ! check_no_admits "$f"; then
        admit_score=0; break
    fi
done
record_score "$TASK_NAME" "no-admits" "$admit_score" 7

# Verification (15 points)
verify_score=0
all_ok=true
for f in "$INTF" "$SPEC" "$IMPL"; do
    if [ -f "$f" ]; then
        if verify_fst "$f" --include "$WORKSPACE/spec" --include "$WORKSPACE/impl" \
            > "$RESULTS_DIR/${TASK_NAME}.$(basename "$f").out" 2>&1; then
            echo -e "${GREEN}  $(basename "$f") OK${NC}"
        else
            echo -e "${RED}  $(basename "$f") FAILED${NC}"
            all_ok=false
        fi
    fi
done
$all_ok && verify_score=15
record_score "$TASK_NAME" "verification" "$verify_score" 15

# Machine-width types in impl (5 points)
mach_score=0
if [ -f "$IMPL" ]; then
    if grep -qE 'UInt64\.t\|UInt32\.t\|SizeT\.t' "$IMPL"; then
        mach_score=3
    fi
    if grep -qE 'Ghost\.erased\|erased' "$IMPL"; then
        mach_score=$((mach_score + 2))
    fi
fi
record_score "$TASK_NAME" "machine-types" "$mach_score" 5

# Makefile quality (5 points)
mk_score=0
if [ -f "$MKFILE" ]; then
    grep -qE 'verify' "$MKFILE" && mk_score=$((mk_score + 1))
    grep -qE 'extract|codegen|krml' "$MKFILE" && mk_score=$((mk_score + 2))
    grep -qE 'bundle' "$MKFILE" && mk_score=$((mk_score + 2))
fi
record_score "$TASK_NAME" "makefile" "$mk_score" 5

# rlimit (5 points)
max_rl=0
for f in "$SPEC" "$IMPL"; do
    [ -f "$f" ] && { m=$(max_rlimit "$f"); (( m > max_rl )) && max_rl=$m; }
done
rlimit_score=0
(( max_rl <= 10 )) && rlimit_score=5 || { (( max_rl <= 50 )) && rlimit_score=3; }
record_score "$TASK_NAME" "rlimit" "$rlimit_score" 5
