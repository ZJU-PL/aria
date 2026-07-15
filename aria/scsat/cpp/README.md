# C++ Shared-Context Batched Satisfiability Helpers

This directory contains a C++ implementation of the shared-context batched
satisfiability algorithms from `aria.scsat.cores`, written against the Z3 C++ API.

## Provided targets
- `scsat_app` – static library with the core algorithms (`scsat.hpp` / `scsat.cpp`)
- `scsat_example` – small driver that mirrors the Python tests and prints results

## Building (CMake)
1. Install Z3 with C++ libraries (`z3++`). On macOS you can use `brew install z3`.
2. From this directory:
   ```bash
   cmake -S . -B build
   cmake --build build
   ./build/scsat_example
   ```
   If CMake cannot find Z3 automatically, set `-DZ3_DIR=/path/to/z3/cmake`.

The example prints result vectors using the same encoding as the Python code:
`sat(1)`, `unsat(0)`, and `unknown(2)`.
