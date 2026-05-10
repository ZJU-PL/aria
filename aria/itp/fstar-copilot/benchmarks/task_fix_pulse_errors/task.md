# Task: Fix Pulse Errors

The file `BrokenPulse.fst` contains a Pulse module with several errors.
Diagnose and fix all errors so the file verifies cleanly.

## Input

The file `BrokenPulse.fst` is provided in your working directory. It contains
a small Pulse library for a bounded buffer (ring buffer). The function
signatures (pre/postconditions) are correct, but the implementations have
bugs that prevent verification.

## Requirements

- Fix all functions so the file verifies with `fstar.exe`
- Do NOT change the function signatures (pre/postconditions)
- No `admit()` or `assume`
- Preserve the overall algorithm logic

## Output

Place the fixed file in: `BrokenPulse.fst` (overwrite the input)
