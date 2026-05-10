# Task: Strengthen Weak Specifications

The files `WeakSpec.fst` and `WeakSpec.fsti` implement a simple key-value
store in Pulse with **weak specifications** — the postconditions only prove
type safety, not functional correctness. Your job is to strengthen the
specifications so they prove full functional correctness.

## Input

- `WeakSpec.fsti` — Interface with weak postconditions
- `WeakSpec.fst` — Implementation with weak proofs

The module implements:
- `create`: Create an empty store
- `insert`: Insert a key-value pair
- `lookup`: Look up a value by key
- `delete`: Delete a key
- `size`: Return the number of entries

## Requirements

1. **Add a pure specification model** — Define a logical model of the store
   (e.g., using `FStar.Map` or a list of pairs) and connect each operation's
   postcondition to it.

2. **Strengthen postconditions** — Each function must have a postcondition
   that references the spec model. For example:
   - `insert`: After inserting `(k,v)`, the model contains `k` mapping to `v`
   - `lookup`: Returns `Some v` iff the model maps `k` to `v`
   - `delete`: The model no longer contains `k`
   - `size`: Returns the number of keys in the model

3. **Update the interface** — `WeakSpec.fsti` must expose the strengthened
   postconditions so callers can reason about the store.

4. **All proofs must verify** — No admits or assumes.

5. You may use the `specreview` skill to analyze the current specs and
   identify weaknesses.

## Constraints

- Preserve the function signatures (parameter types and names)
- No `admit()` or `assume`
- Target rlimit ≤ 10
- The spec model can use unbounded types (`nat`, `list`, `Map`, etc.) since
  it will be erased

## Output

Overwrite the files in place:
- `WeakSpec.fst`
- `WeakSpec.fsti`

Optionally create a new spec module: `StoreSpec.fst`
