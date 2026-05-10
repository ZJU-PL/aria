---
name: projectsetup
description: Structure a new F*/Pulse verification project with Makefile and directory layout
---

## Invocation

This skill is used when:
- Starting a new F*/Pulse verification project from scratch
- Setting up a Makefile for F* verification and C extraction
- Organizing spec vs implementation modules
- Adding OCaml or C testing to a verification project

For building the F*/Pulse/KaRaMeL toolchain itself, see the `sourcebuild` skill.
This skill assumes the toolchain is already built.

## Project Directory Layout

```
myproject/
├── tools/
│   └── FStar/              # fstar2 checkout (gitignored)
├── src/
│   ├── spec/               # Pure specifications
│   │   ├── Types.fst       # Abstract types (may use int, nat, Seq, list)
│   │   └── Spec.fst        # Pure reference implementation
│   └── impl/               # Verified implementations
│       ├── LowTypes.fst    # Machine-width types (UInt64.t, etc.)
│       ├── LowTypes.fsti   # Interface: controls extraction
│       ├── Helpers.fst      # inline_for_extraction utilities
│       ├── Helpers.fsti
│       ├── Impl.fst        # Main implementation (#lang-pulse)
│       └── Impl.fsti       # Public API for extraction
├── test/
│   ├── Test.Spec.fst       # OCaml spec tests
│   └── test_impl.c         # C tests for extracted code
├── snapshot/               # Committed extraction baseline
│   ├── Output.c
│   ├── Output.h
│   └── Makefile            # Standalone build (no F* needed)
├── _cache/                 # Gitignored: .checked files
├── _output/                # Gitignored: .krml files
├── _extract/               # Gitignored: generated .c/.h
├── setup.sh                # Builds toolchain (see sourcebuild skill)
├── Makefile
└── .gitignore
```

## .gitignore

```
tools/FStar
_cache/
_output/
_extract/
*.krml
.depend
```

## Makefile Template

```makefile
# Expects FSTAR_HOME to point to a built FStarLang/FStar@fstar2 checkout
FSTAR_HOME ?= tools/FStar
FSTAR_EXE  ?= $(FSTAR_HOME)/bin/fstar.exe
KRML_HOME  ?= $(FSTAR_HOME)/karamel
KRML_EXE   ?= $(KRML_HOME)/krml

CACHE_DIR  = _cache
OUTPUT_DIR = _output
EXTRACT_DIR = _extract

# F* flags
# --already_cached: skip re-checking the standard libraries
# --ext optimize_let_vc: faster VC generation
# --ext fly_deps: lightweight dependency analysis
FSTAR_FLAGS = --cache_checked_modules \
              --cache_dir $(CACHE_DIR) \
              --odir $(OUTPUT_DIR) \
              --already_cached Prims,FStar,Pulse.Nolib,Pulse.Lib,Pulse.Class,PulseCore \
              --ext optimize_let_vc \
              --ext fly_deps \
              --include src/spec \
              --include src/impl

FSTAR = $(FSTAR_EXE) $(FSTAR_FLAGS)

# Source files
SPEC_FILES  = $(wildcard src/spec/*.fst src/spec/*.fsti)
IMPL_FILES  = $(wildcard src/impl/*.fst src/impl/*.fsti)
ALL_FILES   = $(SPEC_FILES) $(IMPL_FILES)

# ── Dependency analysis ────────────────────────────────────────────
.depend: $(ALL_FILES)
	$(FSTAR) --dep full $(ALL_FILES) --output_deps_to $@

include .depend

# ── Verification ───────────────────────────────────────────────────
$(CACHE_DIR)/%.checked: | $(CACHE_DIR)
	$(FSTAR) $<

$(CACHE_DIR) $(OUTPUT_DIR) $(EXTRACT_DIR):
	mkdir -p $@

verify: $(ALL_CHECKED_FILES)

# ── Extraction (adapt for your modules) ────────────────────────────
# List the modules to extract to .krml
KRML_MODULES = MyProject.LowTypes MyProject.Helpers MyProject.Impl

KRML_FILES = $(patsubst %,$(OUTPUT_DIR)/%.krml,$(subst .,_,$(KRML_MODULES)))

$(OUTPUT_DIR)/%.krml: verify | $(OUTPUT_DIR)
	$(FSTAR) --codegen krml --extract_module $(subst _,.,$*) \
	  src/impl/$(subst _,.,$*).fst

extract-krml: $(KRML_FILES)

# If your code uses tuples (fst/snd), also extract FStar.Pervasives.Native
STDLIB_KRML = $(OUTPUT_DIR)/FStar_Pervasives_Native.krml
$(STDLIB_KRML): verify | $(OUTPUT_DIR)
	$(FSTAR_EXE) --codegen krml --extract_module FStar.Pervasives.Native \
	  --odir $(OUTPUT_DIR) --cache_dir $(CACHE_DIR) \
	  --already_cached Prims,FStar \
	  FStar.Pervasives.Native.fst

extract-c: extract-krml $(STDLIB_KRML) | $(EXTRACT_DIR)
	$(KRML_EXE) \
	  -tmpdir $(EXTRACT_DIR) \
	  -skip-compilation \
	  -warn-error -2-9-17 \
	  -bundle 'MyProject.Impl=MyProject.Impl,MyProject.LowTypes,MyProject.Helpers[rename=Output]' \
	  -bundle 'FStar.*,Pulse.*,PulseCore.*,Prims,MyProject.Types,MyProject.Spec' \
	  -no-prefix MyProject.Impl \
	  $(KRML_FILES) $(STDLIB_KRML)

# ── Testing ────────────────────────────────────────────────────────
test-extracted: extract-c
	$(CC) -I$(KRML_HOME)/include -I$(KRML_HOME)/krmllib/dist/minimal \
	  -I$(EXTRACT_DIR) $(EXTRACT_DIR)/Output.c test/test_impl.c \
	  -o test/test_impl && test/test_impl

# ── Snapshot ───────────────────────────────────────────────────────
update-snapshot: extract-c
	cp $(EXTRACT_DIR)/Output.c snapshot/
	cp $(EXTRACT_DIR)/Output.h snapshot/

.PHONY: verify extract-krml extract-c test-extracted update-snapshot
```

## Module Organization Principles

### Spec vs Implementation Separation

- **`src/spec/`** — Pure specifications using unbounded types (`nat`, `Seq.seq`, `list`,
  `option`). These define "what" the code should do. They are bundled away during C
  extraction and produce no C output.

- **`src/impl/`** — Verified implementations using machine-width types (`UInt64.t`,
  `SizeT.t`, `bool`). These define "how" and are extracted to C.

### Interface Files (.fsti)

Every implementation module that should be visible to other modules (or appear in
the extracted C header) needs an `.fsti` file. Internal helpers in the `.fst` without
`.fsti` declarations become `static` in extracted C.

For detailed guidance on writing extraction-ready code (machine-width types, ghost/erased
parameters, `inline_for_extraction`, avoiding polymorphic stdlib), see the
`krmlextraction` skill.

## OCaml Spec Testing

Extract spec modules to OCaml and test the pure logic before writing Pulse code:

```makefile
FSTAR_LIB_DIR ?= $(FSTAR_HOME)/out/lib
OCAML_DIR = _ocaml

test-ocaml: verify | $(OCAML_DIR)
	$(FSTAR) --codegen OCaml --extract_module MyProject.Types src/spec/Types.fst
	$(FSTAR) --codegen OCaml --extract_module MyProject.Spec src/spec/Spec.fst
	$(FSTAR) --codegen OCaml --extract_module Test.Spec test/Test.Spec.fst
	cd $(OCAML_DIR) && \
	  OCAMLPATH=$(abspath $(FSTAR_LIB_DIR)):$$OCAMLPATH \
	  ocamlfind ocamlopt -package fstar.lib -linkpkg \
	    MyProject_Types.ml MyProject_Spec.ml Test_Spec.ml -o test_spec
	$(OCAML_DIR)/test_spec
```

## Snapshot Pattern

Keep a committed copy of extracted C for users who don't have F*:

```
snapshot/
├── Output.c              # Extracted C source
├── Output.h              # Public header
├── internal/Output.h     # Internal header (if generated)
├── test_impl.c           # Test harness (copied from test/)
├── Makefile              # Standalone build
└── stubs.c               # Any needed stubs (e.g., krmlinit)
```

Snapshot `Makefile`:
```makefile
KRML_HOME ?= path/to/karamel

CFLAGS = -I$(KRML_HOME)/include -I$(KRML_HOME)/krmllib/dist/minimal -I. -Iinternal

test: test_impl.c Output.c stubs.c
	$(CC) $(CFLAGS) $^ -o test_impl && ./test_impl
```

Update snapshot with `make update-snapshot` after any extraction change.

## Additional Resources

- See `sourcebuild` skill for building the F*/Pulse/KaRaMeL toolchain
- See `krmlextraction` skill for KaRaMeL bundle syntax and extraction details
- See `fstarverifier` skill for F*/Pulse error interpretation
