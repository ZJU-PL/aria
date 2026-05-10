---
name: fstarverifier
description: Verify F* and Pulse code with fstar.exe and interpret errors
---

## Invocation

This skill is used when:
- Verifying F* (.fst) or Pulse (`#lang-pulse`) files
- Interpreting F* or Pulse error messages
- Debugging verification or separation logic failures
- Managing Pulse resources (fold/unfold, permissions, memory)

## Verification Commands

```bash
# Verify a single file (stage3 fstar.exe includes Pulse support)
fstar.exe Module.fst

# With project include paths and caching
fstar.exe --cache_checked_modules --cache_dir _cache \
  --already_cached Prims,FStar,Pulse.Nolib,Pulse.Lib,Pulse.Class,PulseCore \
  --include path/to/spec --include path/to/impl \
  Module.fst

# Verify interface first, then implementation (always in this order)
fstar.exe Module.fsti
fstar.exe Module.fst
```

### Diagnostic Flags

| Flag | Purpose |
|------|---------|
| `--query_stats` | Show per-query timing and success/failure |
| `--split_queries always` | Send each assertion as a separate Z3 query |
| `--log_queries` | Write `.smt2` files for Z3 query inspection |
| `--z3refresh` | Restart Z3 between queries (detect flaky proofs) |
| `--print_full_names` | Show fully qualified names (catch symbol confusion) |
| `--print_implicits` | Show implicit arguments (debug unification) |
| `--detail_errors` | More precise error locations, but can take much longer |

```bash
# Combined debugging
fstar.exe --query_stats --split_queries always --z3refresh Module.fst
```

### Resource Limit Options (in-file)

```fstar
#push-options "--z3rlimit 10"        // SMT timeout (target ≤ 10)
#push-options "--fuel 1 --ifuel 1"   // Recursion unfolding depth
#push-options "--z3rlimit 10 --fuel 0 --ifuel 0"  // Tight: no unfolding
```

## Error Interpretation

### "Could not prove post-condition"

**Cause:** SMT cannot establish the postcondition from available facts.

**Solutions:**
1. Add intermediate `assert` statements to locate the gap
2. Call relevant lemmas explicitly
3. Use `Seq.equal` / `Set.equal` for collection equality (not `==`)
4. Call `FS.all_finite_set_facts_lemma()` before FiniteSet reasoning
5. Check that the right definitions are in scope (`--print_full_names`)
6. For Pulse: check fold/unfold balance — every `unfold` needs a matching `fold`

### "Identifier not found: X"

**Cause:** Symbol not in scope.

**Solutions:**
1. Check `open` declarations and `module X = ...` aliases
2. F* is order-sensitive — definitions must precede their use
3. Check for typos; use `--print_full_names` on a working reference

### "rlimit exhausted" / "Query cancelled"

**Cause:** Proof too complex for SMT within the time limit.

**Solutions:**
1. Factor proof into smaller lemmas (most effective)
2. Add intermediate assertions as stepping stones
3. Reduce fuel: `--fuel 0 --ifuel 0`
4. Add explicit type annotations
5. Use `{:pattern ...}` on quantifiers for controlled instantiation
6. Make definitions `[@@"opaque_to_smt"]` and `reveal_opaque` manually

**Do not** just increase rlimit — find the root cause instead.

### "Expected type X, got type Y"

**Cause:** Type mismatch, often involving refinements.

**Solutions:**
1. Add explicit type annotations: `(x <: refined_type)`
2. Check refinement predicates match
3. For machine integers, ensure bounds are established

### "Subtyping check failed" / "Not a subtype of the expected type"

**Cause:** Cannot prove a refinement type's predicate.

**Solutions:**
1. Add an `assert` establishing the predicate just before the expression
2. Call a lemma that establishes the needed fact
3. Check all branches of match/if return the correct type

### "Patterns are incomplete"

**Cause:** Match expression doesn't cover all cases.

**Solutions:**
1. Add missing cases
2. If intentional, add a wildcard `| _ -> ...`
3. Suppress with `--warn_error -321` only if completeness is verified

## Pulse-Specific Errors

### "Application of stateful computation cannot have ghost effect"

**Cause:** Calling a stateful (`stt`) function inside a ghost context.

**How this happens:**
- Variables bound with `with x y. _` are ghost
- If an `if` condition depends on ghost values, both branches become ghost
- Stateful operations (read, write, array access) cannot be ghost

**Solutions:**
1. Read from actual data structures, not ghost witnesses:
```pulse
// WRONG: ghost_seq is ghost from 'with'
let val = Seq.index ghost_seq idx;
let data = !some_ref;  // Error: ghost context

// RIGHT: Read from the actual array
let val = arr.(idx);   // Concrete
```
2. Perform stateful work before entering ghost conditionals
3. Restructure to separate the stateful read from the ghost reasoning

### "Expected a term with non-informative (erased) type"

**Cause:** Trying to bind a concrete type from a ghost expression.

**Solutions:**
1. Keep ghost values ghost: `let x : erased (list entry) = ...`
2. Use assertions instead of bindings: `assert (pure (Cons? ghost_list))`
3. Read concrete data from actual data structures, not ghost state

### "Ill-typed application" in fold/unfold

**Cause:** Predicate arguments don't match the definition.

**Solutions:**
1. Check all arguments match the predicate signature
2. Add explicit type annotations to implicit arguments
3. Verify the predicate definition hasn't changed

### "Cannot prove pure fact"

**Cause:** A `pure (...)` assertion in the slprop cannot be established.

**Solutions:**
1. Add intermediate `assert (pure (...))` steps
2. Call F* lemmas to establish the needed fact
3. Check arithmetic bounds and machine integer properties

## Pulse Resource Management

### Fold/Unfold Balance

Every predicate manipulation must be balanced:

```pulse
unfold (is_valid table spec);
// ... work with exposed resources ...
fold (is_valid table spec);
```

For range predicates, use get/put helpers:
```pulse
get_at ptrs contents lo hi idx;   // Extract element from range
// ... work with element ...
put_at ptrs contents lo hi idx;   // Restore range
```

### Memory Safety Rules

- **Never `drop_` non-empty resources** — this is a memory leak
- **Acceptable drops**: Empty/null/ghost resources only
  ```pulse
  drop_ (LL.is_list null_ptr []);  // OK: empty list is null
  // drop_ (LL.is_list ptr (hd::tl));  // WRONG: memory leak!
  ```
- **Box allocations** need `B.free`: `let b = B.alloc v; ... B.free b`
- **Array resources** must be returned or freed

### Permissions

```pulse
arr |-> contents            // Full permission: read and write
A.pts_to arr #p contents   // Fractional: read-only (p is a fraction)
```

## Common Proof Patterns

### FiniteSet Reasoning
```fstar
// MUST call before FiniteSet assertions
FS.all_finite_set_facts_lemma();
assert (FS.cardinality (FS.remove x s) == FS.cardinality s - 1);
```

### Extensional Equality
```fstar
assert (Seq.equal s1 s2);  // NOT: s1 == s2
assert (Set.equal set1 set2);
```

### Machine Integer Bounds (Pulse)
```pulse
assert (pure (SZ.v idx < len));
assert (pure (len <= SZ.v capacity));
assert (pure (SZ.fits (SZ.v capacity)));
assert (pure (SZ.fits (SZ.v idx + 1)));
let next = idx `SZ.add` 1sz;
```

### Calling F* Lemmas from Pulse
```pulse
my_arithmetic_lemma arg1 arg2;  // Ghost: costs nothing at runtime
assert (pure (conclusion_of_lemma));
```

### Loop Invariants (Pulse)
```pulse
while (!i <^ len)
invariant exists* vi v_acc.
  R.pts_to i vi **
  R.pts_to acc v_acc **
  A.pts_to arr #p s **
  pure (SZ.v vi <= Seq.length s /\ v_acc == partial_result s (SZ.v vi))
{
  // loop body
}
```

**Do NOT use `invariant b. exists* ...`** style.

## Verification Strategy

### For New Code
1. Write the `.fsti` interface first with full pre/post conditions
2. Verify the `.fsti`
3. Implement the `.fst` with `admit()` placeholders to validate structure
4. Remove admits one at a time, adding lemmas as needed
5. Reduce rlimits and harden

### For Failing Proofs
1. Run with `--query_stats` to find slow/cancelled queries
2. Use `--split_queries always` to isolate which assertion fails
3. Add `assert` statements to binary-search the failure point
4. Factor the failing part into a separate lemma
5. If the lemma fails, simplify to a minimal reproducer

### For Flaky Proofs
1. Run with `--z3refresh` to detect order-dependent proofs
2. Reduce rlimit to 10 — if it fails, the proof needs work
3. Add explicit intermediate assertions to guide Z3

## Verification Checklist

- [ ] No `admit()` or `assume_` calls
- [ ] No `drop_` of non-empty resources (Pulse)
- [ ] Interface (.fsti) verified before implementation (.fst)
- [ ] All fold/unfold balanced (Pulse)
- [ ] rlimits ≤ 10 throughout
- [ ] `--query_stats` shows no cancelled queries

## Additional Resources

- [Proof-oriented Programming in F*](https://github.com/FStarLang/PoP-in-FStar)
- `FSTAR_HOME/ulib/` — F* standard library sources
- `FSTAR_HOME/pulse/test/` — Pulse test cases and examples
- `FSTAR_HOME/pulse/lib/pulse/lib/` — Pulse library sources
- See the `proofdebugging` skill for systematic debugging workflows
