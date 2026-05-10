# Task: Verified List Reverse in Pure F*

Write a pure F* module `ListReverse.fst` that implements and verifies list
reversal.

## Requirements

1. Define a **pure specification** function:
   ```
   val reverse_spec : list 'a -> list 'a
   ```
   This is the mathematical definition of reverse (can use any approach:
   accumulator, append, etc.).

2. Define an **efficient tail-recursive implementation**:
   ```
   val reverse : list 'a -> list 'a
   ```
   using an accumulator for O(n) performance.

3. Prove that `reverse` equals `reverse_spec` for all inputs:
   ```
   val reverse_correct : l:list 'a -> Lemma (reverse l == reverse_spec l)
   ```

4. Prove the **involution property**:
   ```
   val reverse_involutive : l:list 'a -> Lemma (reverse (reverse l) == l)
   ```

5. Prove that reverse **preserves length**:
   ```
   val reverse_length : l:list 'a -> Lemma (List.Tot.length (reverse l) == List.Tot.length l)
   ```

6. Prove that reverse **preserves membership**:
   ```
   val reverse_mem : l:list 'a -> x:'a -> Lemma (List.Tot.mem x (reverse l) == List.Tot.mem x l)
   ```

## Constraints

- No `admit()` or `assume`
- All proofs must be machine-checked by `fstar.exe`
- Target rlimit ≤ 10
- The module should be self-contained (only depend on F* standard library)

## Output

Place all code in a single file: `ListReverse.fst`
