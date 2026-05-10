# Task: Verified Array Swap in Pulse

Implement a verified in-place swap of two array elements in Pulse.

## Requirements

1. Define a **pure specification** for swap on sequences:
   ```fstar
   val swap_spec : s:Seq.seq 'a -> i:nat{i < Seq.length s} -> j:nat{j < Seq.length s} ->
     s':Seq.seq 'a{
       Seq.length s' == Seq.length s /\
       Seq.index s' i == Seq.index s j /\
       Seq.index s' j == Seq.index s i /\
       (forall (k:nat). k < Seq.length s /\ k <> i /\ k <> j ==> Seq.index s' k == Seq.index s k)
     }
   ```

2. Implement a **Pulse function** that swaps elements in-place:
   ```pulse
   fn swap (arr: A.array int) (i j: SZ.t)
     (#s: erased (Seq.seq int))
   requires A.pts_to arr s **
     pure (SZ.v i < Seq.length s /\ SZ.v j < Seq.length s)
   ensures exists* s'.
     A.pts_to arr s' **
     pure (s' == swap_spec s (SZ.v i) (SZ.v j))
   ```

3. The implementation should:
   - Read `arr[i]` into a temporary
   - Read `arr[j]` and write it to `arr[i]`
   - Write the temporary to `arr[j]`

4. Prove that after the swap, the array contents match `swap_spec`.

## Constraints

- No `admit()` or `assume`
- Use `#lang-pulse` and `open Pulse.Lib.Pervasives`
- Target rlimit ≤ 10

## Output

Place code in: `Swap.fst`
