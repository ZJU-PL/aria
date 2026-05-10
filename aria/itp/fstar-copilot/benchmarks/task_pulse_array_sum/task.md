# Task: Verified Array Summation in Pulse

Implement a verified imperative array summation in Pulse, with a pure
specification and full functional correctness proof.

## Requirements

1. Define a **pure specification** for summation:
   ```fstar
   let rec sum_spec (s: Seq.seq int) (lo hi: nat) : Tot int (decreases hi - lo) =
     if lo >= hi then 0
     else Seq.index s lo + sum_spec s (lo + 1) hi
   ```

2. Implement a Pulse function that sums an array:
   ```pulse
   fn array_sum (arr: A.array int) (len: SZ.t)
     (#s: erased (Seq.seq int))
   requires A.pts_to arr s ** pure (SZ.v len == Seq.length s)
   returns r: int
   ensures A.pts_to arr s ** pure (r == sum_spec s 0 (Seq.length s))
   ```

3. The implementation must use a **while loop** with:
   - A mutable index `i` iterating from 0 to `len`
   - A mutable accumulator `acc`
   - A loop invariant stating `acc == sum_spec s 0 (SZ.v i)`

4. Prove a helper lemma for the inductive step:
   ```fstar
   val sum_spec_step : s:Seq.seq int -> lo:nat -> hi:nat{lo < hi /\ hi <= Seq.length s} ->
     Lemma (sum_spec s lo hi == sum_spec s lo (hi - 1) + Seq.index s (hi - 1))
   ```
   (or an equivalent formulation that the loop body needs).

## Constraints

- No `admit()` or `assume`
- Use `#lang-pulse` with `open Pulse.Lib.Pervasives`
- Use `SZ.t` for the index and length (machine-width size type)
- Target rlimit ≤ 10
- Array permission: `A.pts_to arr s` (read-only is fine: `A.pts_to arr #p s`)

## Output

Place code in: `ArraySum.fst`
