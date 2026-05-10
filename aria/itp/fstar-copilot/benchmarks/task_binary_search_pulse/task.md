# Task: Verified Binary Search in Pulse

Implement a verified binary search over a sorted array of integers in Pulse.

## Requirements

1. Create a **pure specification** in F* that defines correctness:
   ```
   val binary_search_spec : s:Seq.seq int -> key:int -> option nat
   ```
   Returns `Some idx` if `Seq.index s idx == key`, or `None` if `key` is not in `s`.

2. Implement **binary search** as a Pulse function:
   ```pulse
   fn binary_search (arr: A.array int) (len: SZ.t) (key: int)
     (#s: erased (Seq.seq int))
   requires A.pts_to arr s ** pure (sorted s /\ SZ.v len == Seq.length s)
   returns r: SZ.t
   ensures A.pts_to arr s ** pure (search_postcondition s key r)
   ```
   where `search_postcondition` states:
   - If `r < len`, then `Seq.index s (SZ.v r) == key`
   - If `r == len`, then `key` is not in `s` (i.e., `forall i. i < Seq.length s ==> Seq.index s i <> key`)

3. Define `sorted` as a predicate on sequences:
   ```
   let sorted (s: Seq.seq int) = forall (i j: nat). i <= j /\ j < Seq.length s ==> Seq.index s i <= Seq.index s j
   ```

4. The implementation must use the **standard binary search algorithm**
   with `lo` and `hi` bounds, narrowing by comparing the midpoint.

5. Prove a **loop invariant** that maintains:
   - `lo <= hi <= len`
   - If `key` is in the array, its index is in `[lo, hi)`

## Constraints

- No `admit()` or `assume`
- The file should use `#lang-pulse` and `open Pulse.Lib.Pervasives`
- Target rlimit ≤ 10
- Pulse arrays: use `A.pts_to arr s` for the array permission

## Output

Place code in: `BinarySearch.fst`
