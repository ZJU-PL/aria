---
name: specreview
description: Review F*/Pulse specifications for completeness, strength, and usability
---

## Invocation

This skill is used when:
- Reviewing the quality of F* or Pulse specifications
- Checking that postconditions prove functional correctness, not just type safety
- Auditing a module's interface (.fsti) for completeness
- Verifying that specifications connect implementations to their mathematical models

## Functional Correctness Checklist

For each function with a specification, verify:

### 1. Does the postcondition prove the RIGHT thing?

❌ Weak: `ensures (result >= 0)` — proves a type property, not correctness
✅ Strong: `ensures (result == factorial n)` — proves functional correctness

❌ Weak: `ensures (is_valid_flow result)` — proves a structural property
✅ Strong: `ensures (flow_value result == max_flow graph)` — proves optimality

### 2. Does the postcondition connect to the pure specification?

Every imperative implementation should have a postcondition referencing the
corresponding pure spec function:

```pulse
fn sort_impl (arr: array int) (n: SizeT.t)
  (#s: erased (Seq.seq int))
requires A.pts_to arr s ** pure (SZ.v n == Seq.length s)
ensures exists* s'.
  A.pts_to arr s' **
  pure (
    Seq.length s' == Seq.length s /\    // same length
    sorted s' /\                         // result is sorted
    permutation s s'                     // result is a permutation of input
  )
```

NOT just: `ensures exists* s'. A.pts_to arr s' ** pure (sorted s')`
(This doesn't prove the output is related to the input!)

### 3. Is the specification exposed in the interface?

Check the `.fsti` file:
- All postconditions referencing spec functions must be visible
- If the `.fsti` hides the postcondition, callers cannot reason about the result
- Internal helper predicates used in postconditions should be exported or
  abstracted behind a meaningful interface predicate

```fstar
// BAD .fsti: hides the correctness property
val dijkstra (g: graph) (src: vertex) : ST (array nat) ...
  // No ensures clause visible!

// GOOD .fsti: exposes the spec connection
val dijkstra (g: graph) (src: vertex) : ST (array nat)
  (requires ...)
  (ensures fun dist -> forall v.
    dist.[v] == shortest_path_weight g src v)
```

### 4. Can a caller USE the postcondition?

Verify that the postcondition is stated in terms a caller can work with:
- Uses spec-level types, not implementation-internal predicates
- Quantified properties have useful SMT patterns
- Key lemmas connecting the postcondition to further reasoning are available

### 5. Are algorithm-specific properties proven?

For algorithm implementations, check against the algorithm's mathematical guarantees:

| Algorithm | Must Prove |
|-----------|-----------|
| Sorting | Output is sorted AND a permutation of input |
| Shortest path | Distances are actual shortest paths, not just "some path" |
| MST | Result is a spanning tree AND has minimum weight |
| Search | Returns correct index AND element equality |
| Union-Find | Operations maintain equivalence relation correctly |
| Max-flow | Flow value equals the max-flow (not just "a valid flow") |

## Common Weak-Spec Patterns

### Pattern 1: Proving Type Safety Only

```fstar
// WEAK: proves nothing about what the function computes
val lookup : table -> key -> option value
```

```fstar
// STRONG: proves the lookup matches the logical model
val lookup : t:table -> k:key -> r:option value{r == Table.find (model t) k}
```

### Pattern 2: Proving Partial Properties

```fstar
// WEAK: only proves the output is a tree, not that it's the MST
ensures (is_spanning_tree result graph)
```

```fstar
// STRONG: proves it's minimal among all spanning trees
ensures (is_spanning_tree result graph /\
         forall t. is_spanning_tree t graph ==> weight result <= weight t)
```

### Pattern 3: Postcondition References Impl Internals

```fstar
// WEAK: uses implementation-internal predicate
ensures (imp_valid_state result)

// STRONG: connects to spec via a correspondence lemma
ensures (model result == Spec.expected_state input)
```

### Pattern 4: Missing Permutation/Conservation Properties

```fstar
// WEAK: sorting — doesn't prove output relates to input
ensures (sorted result)

// STRONG: proves it's a sorted permutation of the input
ensures (sorted result /\ permutation_of result input)
```

## Interface Audit Checklist

When reviewing a `.fsti` file:

1. [ ] Every exported function has explicit `requires`/`ensures`
2. [ ] Postconditions reference spec-level definitions, not impl internals
3. [ ] No abstract predicates that hide essential correctness properties
4. [ ] Key lemmas that callers need are exported
5. [ ] Correspondence predicates (relating impl state to spec state) are available
6. [ ] Types that callers need to reason about are not unnecessarily abstract

## Spec Module Completeness

For the pure spec module:
1. [ ] All algorithm operations have pure spec functions
2. [ ] Spec functions match the algorithm's definition (CLRS, paper, etc.)
3. [ ] Key properties of the spec are stated as lemmas
4. [ ] Spec is independent of implementation concerns (no machine integers, no effects)
