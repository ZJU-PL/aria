---
name: krmlextraction
description: Extract verified F*/Pulse code to C via KaRaMeL (.krml intermediate representation)
---

## Invocation

This skill is used when:
- Extracting verified F* or Pulse modules to C code
- Structuring F*/Pulse code so it extracts cleanly
- Configuring KaRaMeL bundle options to control C output layout
- Debugging extraction failures or unexpected C output

## Overview: Two-Phase Pipeline

Extraction from F*/Pulse to C is a two-phase process:

1. **F* → .krml**: F* extracts each module to a KaRaMeL intermediate file (`.krml`)
2. **KaRaMeL → C**: The `krml` tool translates `.krml` files into `.c` and `.h` files

```
 F* source (.fst)          KaRaMeL IR (.krml)           C code (.c/.h)
┌──────────────┐         ┌──────────────────┐         ┌─────────────────┐
│ Module.fst   │─codegen─▶│ Module.krml      │─krml───▶│ Module.c        │
│ Module.fsti  │         │                  │         │ Module.h        │
└──────────────┘         └──────────────────┘         └─────────────────┘
```

## Phase 1: F* to .krml Extraction

### Basic Command

```bash
fstar.exe --codegen krml --extract_module Module.Name \
  --odir _output --cache_dir _cache \
  Module/Name.fst
```

Key flags:
- `--codegen krml` — Produce `.krml` output instead of checking only
- `--extract_module Module.Name` — Extract exactly one module. This produces `Module_Name.krml`
- `--odir _output` — Directory for `.krml` output files
- `--cache_dir _cache` — Directory for `.checked` files from verification

### Why --extract_module (not --extract)

The `--extract 'Module.Name'` flag can pull in submodules (e.g., `Module.Name.Sub`) and
when multiple modules are extracted in a single pass, the output is named `out.krml`
instead of `Module_Name.krml`. Use `--extract_module Module.Name` to extract exactly one
module per invocation with a correctly-named output file.

### Makefile Pattern for Parallel Extraction

```makefile
KRML_MODULES = Foo.Bar Foo.Baz Foo.Impl

KRML_FILES = $(patsubst %,$(OUTPUT_DIR)/%.krml,$(subst .,_,$(KRML_MODULES)))

# Pattern rule: _output/Foo_Bar.krml from Foo.Bar.fst
$(OUTPUT_DIR)/%.krml: verify
	$(FSTAR) --codegen krml --extract_module $(subst _,.,$*) src/$(subst _,.,$*).fst

extract-krml: $(KRML_FILES)
```

This lets `make -j` extract modules in parallel.

### Extracting Standard Library .krml Files

KaRaMeL needs `.krml` definitions for standard library types it encounters (e.g., tuple
projections `fst`/`snd` from `FStar.Pervasives.Native`). Without them, polymorphic
functions like `snd` appear as unresolved calls in the C output.

```bash
# Extract a standard library module
fstar.exe --codegen krml --extract_module FStar.Pervasives.Native \
  --odir _output --cache_dir _cache \
  --already_cached Prims,FStar \
  FStar.Pervasives.Native.fst
```

In a Makefile, define these separately:

```makefile
KRML_STDLIB_MODULES = FStar.Pervasives.Native

KRML_STDLIB_FILES = $(patsubst %,$(OUTPUT_DIR)/%.krml,$(subst .,_,$(KRML_STDLIB_MODULES)))

$(KRML_STDLIB_FILES): verify
	$(FSTAR_EXE) --codegen krml \
	  --extract_module $(subst _,.,$(notdir $(basename $@))) \
	  --odir $(OUTPUT_DIR) --cache_dir $(CACHE_DIR) \
	  --already_cached Prims,FStar \
	  $(subst _,.,$(notdir $(basename $@))).fst
```

Add `$(KRML_STDLIB_FILES)` to the list of `.krml` inputs for the `krml` command.

## Structuring Code for Clean Extraction

### Interface Files (.fsti) Control What Gets Extracted

An `.fsti` file declares the public API of a module. Only declarations in the `.fsti`
are visible to downstream modules and appear in extracted code. Use interfaces to:
- Hide proof-only helpers from extraction
- Control the C function signatures
- Separate spec-level types from implementation types

```
// Foo.fsti — only these are extracted
val bar : UInt64.t -> UInt64.t -> UInt64.t

// Foo.fst — helper is internal, not extracted
let helper x = x + 1
let bar x y = helper x + helper y
```

### Use Machine-Width Integer Types

For code that will be extracted to C, use fixed-width types:
- `UInt64.t`, `UInt32.t`, `UInt16.t`, `UInt8.t` → `uint64_t`, `uint32_t`, etc.
- `SizeT.t` (or `FStar.SizeT.t`) → `size_t`
- `bool` → `bool` (from `<stdbool.h>`)

**Do not use** unbounded types in extractable code:
- `int`, `nat` → No C equivalent (arbitrary precision)
- `list` → No C equivalent (heap-allocated linked list)
- `string` → No C equivalent (F* strings are not C strings)
- `Seq.seq` → No C equivalent (immutable sequences)

These types are fine in:
- Pure spec modules (bundled away, not extracted to C)
- Ghost/erased positions (erased before extraction)
- Proof lemmas (erased entirely)

### Ghost and Erased Types

Use `Ghost.erased` and `ghost` to carry proof-relevant data that vanishes at extraction:

```fstar
// Extracted: only 'table' and 'key' appear in C
fn lookup (table: cap_table)
          (key: UInt64.t)
          (#spec_table: Ghost.erased (Seq.seq entry))  // erased: proof only
  requires pts_to table spec_table
  returns r: UInt64.t
  ensures ...
```

In the extracted C, ghost parameters and their uses disappear entirely.

### inline_for_extraction

Mark small helper functions with `inline_for_extraction` to ensure they are inlined
into callers rather than generating separate C functions:

```fstar
inline_for_extraction
let get_field (w: UInt64.t) (shift width: UInt32.t) : UInt64.t =
  (w `U64.shift_right` shift) `U64.logand` ((U64.sub (U64.shift_left 1UL width) 1UL))
```

This is essential for bitfield accessors, type conversions, and small utilities that
should not be separate function calls in C.

### Proof Lemmas Erase Completely

Functions with `Lemma` return type (or `squash`/`prop` types) produce no C code:

```fstar
val shift_bound_lemma (w: UInt64.t) (s: UInt32.t{v s < 64})
  : Lemma (ensures v (w `shift_right` s) < pow2 (64 - v s))
```

This generates zero C output. Use lemmas freely in proofs without worrying about
extraction overhead.

### Avoid Polymorphic Standard Library Functions in Extracted Code

Polymorphic functions like `FStar.Pervasives.Native.snd` on tuples require KaRaMeL
to have the corresponding `.krml` file for monomorphization. If KaRaMeL cannot
monomorphize them, they appear as unresolved function calls in C.

Either:
1. Extract the needed stdlib `.krml` (e.g., `FStar.Pervasives.Native.krml`) — see above
2. Avoid polymorphic calls: destructure tuples with `let (a, b) = pair in ...` instead of `snd pair`

## Phase 2: KaRaMeL .krml to C

### Basic Command

```bash
krml \
  -tmpdir _extract \
  -skip-compilation \
  -warn-error -2-9-17 \
  -bundle 'Api=Mod1,Mod2,...[rename=OutputName]' \
  -bundle 'FStar.*,Pulse.*,PulseCore.*,Prims' \
  -no-prefix Api.Module \
  input1.krml input2.krml ...
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `-tmpdir DIR` | Output directory for generated `.c` and `.h` files |
| `-skip-compilation` | Generate C only; do not compile or link |
| `-warn-error -W` | Silence warning number W (prefix with `-` to downgrade) |
| `-bundle SPEC` | Group modules into a single C translation unit (see below) |
| `-no-prefix M` | Strip module prefix from M's exported function names |

### Bundle Syntax

The `-bundle` flag controls how F* modules map to C translation units. The full syntax:

```
-bundle 'Api1+Api2=Pattern1,Pattern2,...[rename=Name]'
```

Components:
- **Api modules** (`Api1+Api2`): Modules whose declarations are public (visible in header).
  Multiple API modules are joined with `+`.
- **Pattern list** (`=Pat1,Pat2,...`): All modules matching these patterns are included
  in the bundle. Their non-API declarations become `static`.
- **`[rename=Name]`**: Set the output filename (`Name.c`, `Name.h`).
- **Wildcard patterns**: `FStar.*` matches all FStar submodules.

#### Example: Bundle implementation modules into one C file

```
-bundle 'Impl+Impl.Search=Impl,Impl.Types,Impl.Arith,Helpers[rename=Verified]'
```

This produces:
- `Verified.c` — All code from Impl, Impl.Types, Impl.Arith, Helpers, Impl.Search
- `Verified.h` — Public API: functions from Impl and Impl.Search only
- `internal/Verified.h` — Internal forward declarations

#### Example: Hide spec/proof modules entirely

```
-bundle 'FStar.*,Pulse.*,PulseCore.*,Prims,Spec.Types,Spec.Entry'
```

Modules matching these patterns are bundled together with no API module, so they produce
no C output. This is how you hide pure specification modules from the C build.

### -no-prefix: Clean C Function Names

By default, KaRaMeL prefixes C function names with the module name:
`Caps.Impl.cap_action_from_key` → `Caps_Impl_cap_action_from_key`

Use `-no-prefix Module` to strip the prefix:
`Caps.Impl.cap_action_from_key` → `cap_action_from_key`

Apply to each API module separately:
```
-no-prefix Caps.Impl -no-prefix Caps.Impl.Search
```

### Warning Suppression: Be Careful

KaRaMeL warnings can indicate real problems — do not suppress them blindly. Run `krml`
without `-warn-error` first to see what warnings are emitted, then suppress only those
you understand. Use `-warn-error -W` to downgrade warning W from fatal to non-fatal.

Warnings you may need to suppress for Pulse extraction:

| # | Meaning | Why it fires | Safe to suppress? |
|---|---------|-------------|-------------------|
| 2 | Function not implemented | `Pulse.Lib.Pervasives._zero_for_deref` — a Pulse builtin handled specially by KaRaMeL's C emitter. It has no `.krml` definition but is translated to `*ptr` dereference. | Yes, for this specific Pulse builtin only |
| 9 | Static initializer needed | A global (e.g., a struct constant like `invalid_action`) cannot be a C compile-time constant. KaRaMeL generates `krmlinit.c` with runtime init code. | Yes, if the generated `krmlinit_globals()` is called before use, or the global is only used after main starts |
| 17 | Static initializer declaration | Consequence of warning 9 — the declaration that triggered `krmlinit`. | Yes, same conditions as warning 9 |

Warnings you should **not** suppress without investigation:

| # | Meaning | Risk if suppressed |
|---|---------|-------------------|
| 4 | Type error | Incorrect C types — likely produces wrong code |
| 6 | Variable-length array | VLA in generated C — may crash or be non-portable |
| 11 | Non-Low* expression | Unbounded type leaked into extractable code path |
| 15 | Non-Low* function | Spec function reached extraction — may generate broken C |
| 18 | Bundle collision | Two modules define same symbol — linker errors or wrong behavior |

If warnings 11 or 15 fire, the code likely needs restructuring: move the offending
types/functions behind `Ghost.erased` or into spec-only modules that are bundled away.

## Complete Makefile Template

For a full working Makefile that integrates verification, extraction, testing, and
snapshot management, see the `projectsetup` skill.

## Debugging Extraction Issues

### Unresolved function call in C output

**Symptom**: `undefined reference to FStar_Pervasives_Native_snd` at link time.

**Cause**: KaRaMeL couldn't monomorphize a polymorphic stdlib function because the
corresponding `.krml` file was not provided.

**Fix**: Extract the needed stdlib `.krml` and include it in the `krml` command inputs.

### out.krml instead of Module_Name.krml

**Symptom**: F* produces `out.krml` instead of the expected filename.

**Cause**: `--extract 'Module'` pulled in submodules (`Module.Sub`), extracting
multiple modules in one pass.

**Fix**: Use `--extract_module Module.Name` to extract exactly one module per invocation.

### Empty or missing C functions

**Symptom**: A function you expected in the C output is missing.

**Causes**:
1. The function has `Lemma` return type → erased (correct behavior)
2. The function is in a module matched by the hide-bundle pattern
3. The module's `.fsti` doesn't declare the function
4. The function is ghost/erased

**Fix**: Check bundle patterns; ensure the module is in the API bundle, not the hide bundle.

### Wrong function names in C

**Symptom**: C functions have long prefixed names like `Caps_Impl_cap_action_from_key`.

**Fix**: Add `-no-prefix Caps.Impl` to the `krml` command.

### Type mismatches after extraction

**Symptom**: KaRaMeL reports type errors or generates incorrect C types.

**Cause**: Using unbounded F* types (`int`, `nat`, `list`) in extractable code paths.

**Fix**: Use machine-width types (`UInt64.t`, `UInt32.t`, `SizeT.t`, `bool`) in all
code that reaches extraction. Move unbounded types behind `Ghost.erased` or into
spec-only modules.

## Additional Resources

- [KaRaMeL README](https://github.com/FStarLang/karamel)
- `krml --help` for full flag reference
- [F* tutorial on extraction](https://fstar-lang.org/tutorial/)
