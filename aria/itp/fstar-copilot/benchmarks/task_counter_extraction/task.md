# Task: Verified Counter with C Extraction

Build a verified counter module in Pulse and extract it to C via KaRaMeL.
This exercises the full pipeline: specification, implementation, verification,
and extraction.

## Requirements

### Specification (spec/Counter.Spec.fst)

1. Define a pure counter model:
   ```fstar
   type counter_state = { value: nat; max_value: nat }

   let init_spec (max: nat) : counter_state = { value = 0; max_value = max }
   let increment_spec (s: counter_state) : option counter_state = ...
   let read_spec (s: counter_state) : nat = s.value
   let reset_spec (s: counter_state) : counter_state = { s with value = 0 }
   ```

2. `increment_spec` returns `None` if `value >= max_value`, otherwise
   increments.

### Implementation (impl/Counter.Impl.fst, #lang-pulse)

3. Represent the counter as a Pulse `ref UInt64.t` with a ghost `max` parameter.

4. Define an abstract predicate:
   ```pulse
   let counter_inv (r: ref UInt64.t) (s: erased counter_state) = ...
   ```

5. Implement:
   - `init`: Allocate a counter on the stack
   - `increment`: Increment if below max, return `true`; otherwise return `false`
   - `read`: Read current value
   - `reset`: Set value to 0

6. Each function's postcondition must reference the corresponding `*_spec` function.

### Interface (impl/Counter.Impl.fsti)

7. Export all functions with their full specifications. Hide internal details.

### Extraction

8. All implementation types must use machine-width integers (`UInt64.t`, `bool`).
9. Ghost/erased parameters for spec-level state.
10. Mark small helpers with `inline_for_extraction`.
11. Provide a `Makefile` with:
    - `verify` target: verifies all .fst/.fsti files
    - `extract-c` target: extracts to C via KaRaMeL
    - Correct `-bundle` flags to hide spec modules

## Constraints

- No `admit()` or `assume`
- Target rlimit ≤ 10
- Extracted C must compile with `gcc -c`

## Output

Create this structure:
```
spec/Counter.Spec.fst
impl/Counter.Impl.fst
impl/Counter.Impl.fsti
Makefile
```
