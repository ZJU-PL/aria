# Task: Analyze and Improve an Existing Codebase

Given an existing F*/Pulse module, perform a comprehensive analysis and
improve it across multiple dimensions: specification strength, proof
robustness, code organization, and readability.

## Input

`Unoptimized.fst` is a Pulse module implementing a simple hash set (using
linear probing) with various issues:

1. **Weak specifications** — Postconditions don't prove functional correctness
2. **Monolithic structure** — Everything is in one large file
3. **High rlimits** — Some proofs need rlimit 100+
4. **Missing interface** — No .fsti file
5. **Poor separation** — Spec and impl are interleaved
6. **Admits present** — Two functions have admitted proofs

## Requirements

### Analysis Phase

Use the `specreview` skill to analyze the specifications:
- Identify which postconditions are weak
- Identify missing correctness properties
- Suggest what the specs should prove

Use the `proofdebugging` / `smtprofiling` skills to analyze performance:
- Identify which proofs are slow
- Suggest optimization strategies

### Improvement Phase

1. **Split into modules**:
   - `HashSet.Spec.fst` — Pure hash set model
   - `HashSet.Impl.fst` — Pulse implementation
   - `HashSet.Impl.fsti` — Interface

2. **Strengthen specifications**:
   - Connect each operation to the spec model
   - Prove functional correctness, not just type safety

3. **Fix the admits**:
   - Complete the two admitted proofs

4. **Optimize proof performance**:
   - Reduce all rlimits to ≤ 20

5. **Add extraction readiness**:
   - Use machine-width types in impl
   - Mark helpers as `inline_for_extraction`

## Constraints

- The improved code must verify with `fstar.exe`
- No `admit()` or `assume` in the final result
- Target rlimit ≤ 20 (ideally ≤ 10)
- Preserve the external API (function names and behavior)

## Output

Create all improved files in the workspace.
