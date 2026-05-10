---
name: smtprofiling
description: Debug F* queries sent to Z3, diagnosing proof instability and performance issues
---

## Invocation
This skill is used when:
- Verifying F* (.fst) or interface (.fsti) files
- Debugging verification failures and proof performance
- Especially when proofs require high rlimits or fail unpredictably
- A proof requires rlimit > 20 (auto-invoke this skill rather than escalating)
- A proof that previously passed starts failing (regression)
- Build time exceeds expectations for a module

## Core Operations

### Collect an .smt2 file for problematic proof

Wrap the part of the program to diagnose proof failures with

```fstar
#push-options "--log_queries --z3refresh --query_stats --split_queries always"
let definition_to_be_debugged ...
#pop-options
```

Run F* on the file (with appropriate include paths)

```bash
fstar.exe Module.fst
```

The log messages will show the name of the .smt2 file logged for each proof obligation.

Using `--z3refresh` together with `--log_queries` produces a
self-contained `.smt2` file per query (each query starts with a fresh
Z3 process). This is critical for isolating expensive sub-goals: each
`.smt2` file can be profiled independently.

Using `--split_queries always` decomposes compound VCs into individual
sub-goals. This lets you identify exactly which conjunct of a
proof obligation is expensive. Cross-reference the query number in the
`Query-stats` output with the `.smt2` file number to map each timing
to its isolated file.

### Verify .smt2 file independently of F*

Find `z3` in the path. It might be named `z3-4.13.3`, `z3-4.15.1` etc., with a version number suffix

```bash
z3 queries-myquery.smt2
```

You can add z3 options like `smt.qi.profile=true` to see which quantifiers were firing 
too much and what to do about it

### Interpreting a quantifier profile

The Z3 quantifier profile is printed to stderr. Capture it with:

```bash
z3 smt.qi.profile=true queries-myquery.smt2 2> qi_profile.txt
```

Parse the output to get per-quantifier totals:

```bash
awk '/\[quantifier_instances\]/ {
  name = $2; count = $4; total[name] += count;
} END {
  for (n in total) printf "%8d %s\n", total[n], n;
}' qi_profile.txt | sort -rn | head -20
```

Some quantifiers in F*'s SMT encoding always fire a lot---this is
by design. Typically, quantifiers with unqualified names 
(Box_bool_proj_0) or those with names in Prims, or FStar.
Pervasives will fire a lot.

Look for quantifiers in modules that are in files that you 
authored that are firing a lot---these might signify that you 
should write a pattern for those quantifiers, or control their 
instantiation explicitly.

### Common quantifier cascade patterns

A **quantifier cascade** is when one quantifier instantiation produces
terms that trigger another quantifier, which produces more terms,
creating a positive feedback loop. This is the most common cause of
queries taking minutes instead of seconds.

Signs of a cascade:
- One or two quantifiers have 10,000+ instantiations while everything else has < 100
- Query takes very long wall-clock time but uses modest rlimit
- The dominant quantifier has a single-term pattern like `[SMTPat (f x y)]`
  and its conclusion introduces new terms that match the pattern of
  another quantifier

Example cascade found in EverParse:
1. `cbor_map_equal` unfolds to `forall k. cbor_map_get m1 k == cbor_map_get m2 k`
2. Each `cbor_map_get` produces `cbor_map_defined` terms
3. `cbor_map_defined_alt` has `[SMTPat (cbor_map_defined k f)]` and
   introduces `exists v. cbor_map_mem (k,v) f`
4. Existential skolemization creates new ground terms
5. New terms trigger more pattern matches → 64,000+ instantiations

### Fixing quantifier cascades

**Option 1: Remove the SMTPat and add a `bring_` helper**

Remove `[SMTPat ...]` from the offending lemma. Add a wrapper that
re-introduces the quantifier with its pattern in a controlled scope:

```fstar
// Original: always active, causes cascades
let problematic_lemma (x: t) (y: t)
: Lemma (some_prop x y)
  [SMTPat (f x y)]    // REMOVE THIS
= ...

// New: call bring_ only where needed
let bring_problematic_lemma ()
: Lemma (forall (x: t) (y: t) . {:pattern (f x y)} some_prop x y)
= Classical.forall_intro_2 problematic_lemma
```

Then call `bring_problematic_lemma ()` only in the specific proofs
that need it, rather than polluting all proofs globally.

**Option 2: Use `--using_facts_from` to prune per-function**

Surgically remove a lemma from Z3's context for a specific function
without modifying the lemma itself:

```fstar
#push-options "--using_facts_from '* -Module.Name.problematic_lemma'"
let my_expensive_function ...
#pop-options
```

This is useful for quick experimentation before committing to
Option 1. Verify the quantifier is actually removed by checking the
`.smt2` file.

**Option 3: Use multi-pattern triggers**

Change a single-term pattern to a conjunctive multi-pattern so the
lemma only fires when multiple terms are present:

```fstar
// Before: fires whenever (f x y) appears
[SMTPat (f x y)]

// After: fires only when both (f x y) AND (g x) are present
[SMTPat (f x y); SMTPat (g x)]
```

### Systematic profiling workflow

For large projects, profile the entire build:

```bash
# 1. Full build with --query_stats
make -j$(nproc) OTHERFLAGS="--query_stats" 2>&1 | tee build_profile.log

# 2. Find top files by SMT time
grep -a "Query-stats" build_profile.log | grep "succeeded" \
  | sed -n 's/(\([^(]*\)\.\(fst\|fsti\)([^)]*)).*succeeded in \([0-9]*\) milliseconds.*/\1.\2 \3/p' \
  | awk '{f[$1]+=$2} END {for(k in f) print f[k], k}' | sort -rn | head -20

# 3. Find top individual queries
grep -a "Query-stats" build_profile.log | grep "succeeded" \
  | sed -n 's/.*Query-stats (\([^)]*\)).*succeeded in \([0-9]*\) milliseconds.*rlimit \([0-9]*\) (used rlimit \([0-9.]*\)).*/\2\t\4\t\3\t\1/p' \
  | sort -rn | head -20

# 4. For the most expensive query, isolate it:
#push-options "--log_queries --z3refresh --query_stats --split_queries always"

# 5. Profile the isolated .smt2 file:
z3 smt.qi.profile=true queries-Module-NNN.smt2 2> qi_profile.txt

# 6. Parse the profile to find the dominant quantifier(s)
```

### F* options for controlling SMT performance

| Option | Effect |
|--------|--------|
| `--split_queries always` | Each assertion becomes its own Z3 query |
| `--z3refresh` | Fresh Z3 process per query (no accumulated state) |
| `--log_queries` | Write `.smt2` files for each query |
| `--query_stats` | Print per-query timing and rlimit usage |
| `--z3cliopt smt.arith.nl=false` | Disable non-linear arithmetic |
| `--z3cliopt smt.qi.eager_threshold=N` | Limit quantifier instantiation eagerness |
| `--using_facts_from '* -Name'` | Prune specific lemmas from Z3 context |
| `--ext context_pruning` | Prune unreachable assumptions (default on) |
| `--fuel N` / `--ifuel N` | Control recursive unfolding depth |

### Tightening fuel, ifuel, and rlimit

After fixing expensive queries, tighten the settings:
- Reduce `fuel` and `ifuel` to the minimum that works (prefer 0-2)
- Reduce `rlimit` to 2-4x the `used rlimit` reported by `--query_stats`
- High fuel (>2) causes exponential unfolding; replace with explicit lemma calls
- High ifuel (>2) causes excessive inversion; add explicit match/inversion steps

### Reusing Proof Strategies for Stabilizing Proofs

When stuck on a proof performance problem, search for commits in the same
project that successfully stabilized similar proofs. github.com/FStarLang/AlgoStar 
is a good example of a project with many proof performance commits to mine for techniques.:

Study what techniques were used — factoring lemmas, adding patterns,
using calc proofs, adding `opaque_to_smt`, using `--using_facts_from` — and
apply the same techniques to the current problem.

### Proof Stabilization Catalog

The following catalog is mined from FStarLang/AlgoStar (897 commits of
verified CLRS algorithm implementations in F*/Pulse). These are proven
techniques organized by type of problem.

#### Technique 1: Factor Large Modules Into Smaller Ones

**Problem:** Large `.fst` files make Z3 slow because every function's VC
carries the context of all preceding definitions.

**Solution:** Extract groups of related lemmas into separate modules.

**Case study (Prim MST):**
- `Prim.Impl.fst` was 2267 lines with 76 inline math lemmas — slow and fragile
- Factored into: `Prim.Greedy.fst` (1010 lines, 41s) for all greedy safety
  lemmas; `Prim.KeyInv.fst` (496 lines, 4s) for the key invariant module;
  `Prim.Defs.fst` (68 lines, 1.5s) for basic definitions
- `Prim.Impl.fst` shrank to 938 lines (59% reduction)
- See commits `d1665a6`, `aada386` in FStarLang/AlgoStar

**Case study (Heapsort):**
- Deduplicated `_upto` lemma proofs by having non-`_upto` versions delegate
  to the general versions — eliminated one copy of an expensive
  `perm_prefix_bounded_aux` proof
- `Lemmas.fst`: 238s → 63s (74% reduction)
- See commit `29242fb` in FStarLang/AlgoStar

#### Technique 2: Use `#restart-solver` Between Key Functions

**Problem:** Z3's accumulated state from prior functions pollutes the context
for subsequent proofs, causing slowness or spurious failures.

**Solution:** Place `#restart-solver` between large functions to reset Z3.

**Case study (Heapsort):**
- Added `#restart-solver` before `build_max_heap`
- `Impl.fst`: 445s → 241s (46% reduction)
- See commit `29242fb` in FStarLang/AlgoStar

**Case study (Kruskal NL arithmetic):**
- NL arithmetic (`u * n + v < n * n`) worked in isolation but failed
  in-context due to accumulated Z3 state
- `#restart-solver` between key functions fixed it
- See commit `01e2e8d` in FStarLang/AlgoStar

#### Technique 3: Replace `--z3refresh` with `--split_queries always`

**Problem:** `--z3refresh` starts a new Z3 process per query (expensive);
proofs need high rlimit.

**Solution:** `--split_queries always` decomposes VCs into smaller queries
without the overhead of process restarts, often allowing lower rlimits.

**Case study (CountingSort):**
- Switched from `--z3rlimit 800 --z3refresh` to `--z3rlimit 400 --split_queries always`
- 50% rlimit reduction with same reliability
- See commit `5e35796` in FStarLang/AlgoStar

#### Technique 4: Opaque Bundles with Explicit Instantiation Lemmas

**Problem:** Complex predicates cause Z3 to spend time unfolding definitions
it doesn't need.

**Solution:** Make predicates `[@@"opaque_to_smt"]` and provide explicit
`_at` instantiation lemmas (for `init`, `after_update`, etc.).

**Case study (Prim KeyInv):**
- `key_inv`, `ims_finite_key`, `parent_in_mst` all made opaque
- Explicit `key_inv_init`, `key_inv_after_update`, `key_inv_at` lemmas
  for controlled instantiation
- `parent_in_mst_after_update` uses `introduce forall` + 3-way case split
  with explicit lemma calls — no Z3 quantifier matching needed
- Entire module verifies in 2.6s
- See commit `aada386` in FStarLang/AlgoStar

#### Technique 5: Wrap NL Arithmetic in Helper Functions

**Problem:** Non-linear arithmetic expressions like `u * n + v` cause Z3
to enter expensive NL reasoning.

**Solution:** Bundle NL index computation in a function with refinement types,
then use `FStar.Math.Lemmas` for explicit NL bounds.

**Case study (Kruskal adj_weight):**
- `adj_weight` bundles `u * n + v` index in a refined function type
- `FStar.Math.Lemmas.lemma_mult_lt_right` for explicit NL bounds
- Unblocked a proof that was stuck for days on NL arithmetic
- See commits `bf00c94` → `01e2e8d` in FStarLang/AlgoStar

#### Technique 6: Aggressive rlimit Reduction After Proof Works

**Problem:** Proofs initially written with high rlimits are flaky —
they may break with unrelated changes to the Z3 context.

**Solution:** Once a proof works, systematically reduce rlimit to the
minimum needed (target ≤ 10, tolerate up to 200 for complex Pulse VCs).

**Case study (ch22 BFS/DFS):**
- `queue_bfs`: 2400 → 200 (92% reduction)
- `maybe_discover`: 600 → 200 (67% reduction)
- `dfs_visit`: 800 → 200 (75% reduction)
- See commit `896a145` in FStarLang/AlgoStar

**Case study (ch26 MaxFlow):**
- `max_flow`: 600 → 50 (92% reduction!)
- `find_bottleneck_imp`, `augment_imp`: 200 → 80 (60% reduction)
- See commit `eeb0fdb` in FStarLang/AlgoStar

#### Technique 7: Hoist Heavy Sub-computations Out of Loop Bodies

**Problem:** Large Pulse functions take long to verify, with every proof obligation
going issuing its own Z3 query, and with the assumptions in the context accumulating
through the context of the function.

**Solution:** Extract helpers: factor `fn` for sub-tasks, call them
from the enclosing function, allowing the proof to proceed more modularly.

**Case study (Prim MST):**
- `find_min_vertex`: hoisted to separate fn → 3.5min build (was 4.5min)
- `prim_step`: hoisted to separate fn → cleaner invariant structure
- `update_keys`: hoisted to separate fn → focused VC
- See commits `91e8b43`, `fdbf5bc` in FStarLang/AlgoStar

#### Technique 8: Set Fuel Precisely per Lemma

**Problem:** Fuel 2 causes Z3 to try unfolding recursion two levels deep,
which can be exponentially expensive with multiple recursive definitions.

**Solution:** Set `--fuel 0 --ifuel 0` as the module default, then use
`#push-options "--fuel N"` only on lemmas that need it.

**Case study (ch09 PartialSelectionSort):**
- `remove_element_count_le/lt` needed `--fuel 4` explicitly
- With default fuel 2, Z3 spent 200s on fuel-retry; with explicit fuel 4, 29s
- Full build: 7min → 1min (7× speedup)
- See commit `4c158fa` in FStarLang/AlgoStar

#### Technique 9: Use `introduce forall` Instead of SMT Quantifier Matching

**Problem:** Complex universal quantifiers with tricky patterns — Z3 fails
to find the right instantiation.

**Solution:** Use F*'s `introduce forall` tactical sugar with explicit
case splits and lemma calls, bypassing Z3's pattern matching entirely.

**Case study (Prim parent_in_mst_after_update):**
- Uses `introduce forall` + 3-way case split (v=u, v in MST, v not in MST)
- Each case handled with explicit lemma call
- No SMT pattern needed — proof is fully explicit
- See commit `aada386` in FStarLang/AlgoStar

#### Technique 10: Avoid Wrapper Types in Quantifier Bodies

**Problem:** Using helper functions inside quantifier bodies prevents Z3
from matching patterns.

**Solution:** Use raw expressions (e.g., `Seq.index`) in opaque predicate
bodies rather than convenience wrappers.

**Case study (Prim KeyInv):**
- `key_inv` initially used `swt` (safe weight accessor) in its quantifier
- Changed to raw `w*n+v` / `Seq.index` — matches what `prim_safe_add_vertex` needs
- Entire module verifies in 3s
- See commit `c992174` in FStarLang/AlgoStar

### Quantifier Instantiation Threshold

If Z3 fails to instantiate long quantifier chains (proof fails even though
the required facts are in scope):
- The `smt.qi.eager_threshold` controls how aggressively Z3 instantiates
  quantifiers. Default is 10.
- If confident there are no instantiation loops, try a higher value:
  `--z3smtopt '(set-option :smt.qi.eager_threshold 100)'`
- Symptoms: proof works at high rlimit but fails at low rlimit because Z3
  gives up instantiating before finding the proof
- Diagnosis: check the quantifier profile for quantifiers with exactly
  `eager_threshold` instantiations — these were cut off

### More information on SMT profiling

Find PoP-in-FStar at https://github.com/FStarLang/PoP-in-FStar

See https://fstar-lang.org/tutorial/book/under_the_hood/uth_smt.html#profiling-z3-and-solving-proof-performance-issues

The section under_the_hood/uth_smt.rst contains information about F*'s SMT
encoding and how to profile.
