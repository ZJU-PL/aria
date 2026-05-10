# Task: Verified Insertion Sort with Full Correctness

Implement a fully verified insertion sort in Pulse over mutable arrays, with
pure specifications proving both the sorted property AND the permutation property.

## Requirements

### Specification Module (InsertionSort.Spec.fst)

1. Define `sorted`:
   ```fstar
   let sorted (s: Seq.seq int) =
     forall (i j: nat). i <= j /\ j < Seq.length s ==> Seq.index s i <= Seq.index s j
   ```

2. Define `permutation_of` (multiset equality):
   ```fstar
   val is_permutation_of : Seq.seq int -> Seq.seq int -> prop
   ```
   Two sequences are permutations if every element appears the same number of
   times. You may define a `count` function and compare counts.

3. Prove basic lemmas about permutation (reflexivity, transitivity, that
   swapping two elements produces a permutation).

### Implementation Module (InsertionSort.Impl.fst, #lang-pulse)

4. Implement `insert`:
   ```pulse
   fn insert (arr: A.array int) (pos: SZ.t) (#s: erased (Seq.seq int))
   requires A.pts_to arr s ** pure (...)
   ensures exists* s'. A.pts_to arr s' ** pure (...)
   ```
   Inserts `arr[pos]` into the sorted prefix `arr[0..pos)` by shifting
   elements right.

5. Implement `insertion_sort`:
   ```pulse
   fn insertion_sort (arr: A.array int) (len: SZ.t) (#s: erased (Seq.seq int))
   requires A.pts_to arr s ** pure (SZ.v len == Seq.length s)
   ensures exists* s'.
     A.pts_to arr s' **
     pure (sorted s' /\ is_permutation_of s' s)
   ```

6. Both functions need **loop invariants** connecting the concrete array
   state to the pure specification.

### Interface (InsertionSort.Impl.fsti)

7. Export `insertion_sort` with its full postcondition so callers can reason
   about the result.

## Constraints

- No `admit()` or `assume`
- Postconditions must prove BOTH `sorted` AND `is_permutation_of`
- Target rlimit ≤ 20 (this is a harder problem; 10 is ideal but 20 is acceptable)
- Use `#lang-pulse` for the implementation module

## Output

Create these files:
- `InsertionSort.Spec.fst` — Pure specifications and permutation lemmas
- `InsertionSort.Impl.fst` — Pulse implementation with proofs
- `InsertionSort.Impl.fsti` — Public interface
