# Task: Set Up a New F*/Pulse Verification Project

Given a natural-language description of a project, set up the complete
directory structure, Makefile, and skeleton files for an F*/Pulse verification
project.

## Project Description

Create a project called **"VerifiedQueue"** — a verified FIFO queue data
structure implemented in Pulse with C extraction.

The queue should support:
- `create`: Create an empty queue with a given capacity
- `enqueue`: Add an element to the back (fail if full)
- `dequeue`: Remove and return the front element (fail if empty)
- `peek`: Return the front element without removing it
- `size`: Return the current number of elements
- `is_empty` / `is_full`: Boolean tests

## Requirements

1. **Directory layout** following the `projectsetup` skill:
   ```
   spec/       — Pure specifications
   impl/       — Pulse implementations
   test/       — Test files (at least a test skeleton)
   ```

2. **Makefile** with targets:
   - `verify`: Run fstar.exe on all modules
   - `extract-c`: Extract implementation to C via KaRaMeL
   - `clean`: Remove build artifacts
   - Correct `--already_cached`, `--include`, `--cache_dir` flags
   - Correct `-bundle` flags for extraction

3. **Skeleton source files**:
   - `spec/Queue.Spec.fst` — Pure queue model (using `Seq.seq` or `list`)
   - `impl/Queue.Impl.fst` — Pulse implementation stubs (functions can use `admit()` since this is a skeleton)
   - `impl/Queue.Impl.fsti` — Interface with full pre/postconditions referencing the spec

4. **.gitignore** — Ignore `_cache/`, `_output/`, `_extract/`, `tools/`

5. The skeleton should be **immediately verifiable** — at least the spec
   module and the .fsti should verify (the .fst may have admits as stubs).

## Constraints

- Use `projectsetup` skill conventions
- Use `#lang-pulse` for the implementation
- Use machine-width types in the implementation (`UInt64.t`, `SZ.t`, `bool`)
- Spec module should use unbounded types (`nat`, `Seq.seq`)

## Output

Create all files under the workspace root.
