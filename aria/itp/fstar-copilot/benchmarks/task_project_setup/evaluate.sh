#!/usr/bin/env bash
# Evaluator for task_project_setup
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"

# Directory structure (8 points)
dir_score=0
[ -d "$WORKSPACE/spec" ] && dir_score=$((dir_score + 2))
[ -d "$WORKSPACE/impl" ] && dir_score=$((dir_score + 2))
[ -f "$WORKSPACE/Makefile" ] && dir_score=$((dir_score + 2))
[ -f "$WORKSPACE/.gitignore" ] && dir_score=$((dir_score + 2))
record_score "$TASK_NAME" "directory-structure" "$dir_score" 8

# Source files exist (8 points)
file_score=0
for pat in "spec/*Spec*" "impl/*Impl*.fst" "impl/*Impl*.fsti"; do
    found=$(find "$WORKSPACE" -path "$WORKSPACE/$pat" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        file_score=$((file_score + 2))
    fi
done
# Test directory/files
if [ -d "$WORKSPACE/test" ] || find "$WORKSPACE" -name '*test*' -o -name '*Test*' | grep -q .; then
    file_score=$((file_score + 2))
fi
record_score "$TASK_NAME" "source-files" "$file_score" 8

# Makefile quality (10 points)
mk_score=0
MK="$WORKSPACE/Makefile"
if [ -f "$MK" ]; then
    grep -qE '^verify' "$MK" && mk_score=$((mk_score + 2))
    grep -qE 'extract|codegen' "$MK" && mk_score=$((mk_score + 2))
    grep -qE 'clean' "$MK" && mk_score=$((mk_score + 1))
    grep -qE 'already_cached' "$MK" && mk_score=$((mk_score + 2))
    grep -qE 'bundle' "$MK" && mk_score=$((mk_score + 2))
    grep -qE 'cache_dir' "$MK" && mk_score=$((mk_score + 1))
fi
record_score "$TASK_NAME" "makefile-quality" "$mk_score" 10

# .gitignore quality (4 points)
gi_score=0
GI="$WORKSPACE/.gitignore"
if [ -f "$GI" ]; then
    grep -qE '_cache' "$GI" && gi_score=$((gi_score + 1))
    grep -qE '_output' "$GI" && gi_score=$((gi_score + 1))
    grep -qE '_extract' "$GI" && gi_score=$((gi_score + 1))
    grep -qE 'tools|FStar' "$GI" && gi_score=$((gi_score + 1))
fi
record_score "$TASK_NAME" "gitignore" "$gi_score" 4

# Spec file verifies (5 points)
spec_file=$(find "$WORKSPACE/spec" -name '*.fst' 2>/dev/null | head -1)
spec_score=0
if [ -n "$spec_file" ] && [ -f "$spec_file" ]; then
    if verify_fst "$spec_file" --include "$WORKSPACE/spec" --include "$WORKSPACE/impl" \
        > "$RESULTS_DIR/${TASK_NAME}.spec.out" 2>&1; then
        echo -e "${GREEN}Spec verifies${NC}"
        spec_score=5
    else
        echo -e "${RED}Spec does not verify${NC}"
    fi
fi
record_score "$TASK_NAME" "spec-verifies" "$spec_score" 5

# Pulse code present (5 points)
pulse_score=0
impl_file=$(find "$WORKSPACE/impl" -name '*.fst' 2>/dev/null | head -1)
if [ -n "$impl_file" ] && [ -f "$impl_file" ]; then
    grep -qE '#lang-pulse|Pulse' "$impl_file" && pulse_score=$((pulse_score + 3))
    grep -qE 'UInt64|UInt32|SizeT' "$impl_file" && pulse_score=$((pulse_score + 2))
fi
record_score "$TASK_NAME" "pulse-impl" "$pulse_score" 5
