# Task: Fix Broken F* Proofs

The file `BrokenLemmas.fst` contains several lemmas with incorrect or
incomplete proofs. Your job is to fix all the proofs so the file verifies
cleanly with `fstar.exe`.

## Input

The file `BrokenLemmas.fst` is provided in your working directory. It contains
5 lemmas about list operations. Each lemma's `val` declaration (statement) is
correct, but the proof body is broken — it either fails to verify, is
incomplete, or is admitted.

## Requirements

- Fix all 5 proofs so the file verifies with `fstar.exe`
- Do NOT change the lemma statements (the `val` declarations)
- No `admit()` or `assume` in the final result
- Target rlimit ≤ 10
- You may add helper lemmas if needed

## Output

Place the fixed file in: `BrokenLemmas.fst` (overwrite the input)
