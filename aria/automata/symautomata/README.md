# Automata

# Symbolic Automata

https://github.com/spencerwuwu/symautomata

A python framework for working with Automata. The framework contains a python implementation and C bindings using pywrapfst (optional).
This framework is part of the [lightbulb-framework](https://github.com/lightbulb-framework/lightbulb-framework).

## SFA API

The symbolic finite automata entrypoint is `aria.automata.symautomata.SFA`.

Use `SFA.symbolic(symbolic_symbol=..., symbolic_universe=..., ...)` to create a
symbolic automaton without repeating the symbolic-domain arguments at each
constructor call.

Use `SFA.symbolic_bitvec(width, name=..., symbolic_universe=..., ...)` when the
domain is a bit-vector sort and you want the builder to allocate the symbolic
variable for you.

Use `SFA.symbolic_int(name=..., symbolic_universe=..., ...)` and
`SFA.symbolic_bool(name=..., symbolic_universe=..., ...)` for integer and
Boolean symbolic domains.

- `Predicate` is the abstract guard contract.
- `SetPredicate` is the finite-alphabet guard implementation.
- `Z3Predicate` is the solver-backed guard implementation for symbolic alphabets.
- For symbolic automata, guards are plain formulas and the symbolic domain lives on
  the `SFA` via `symbolic_symbol` and `symbolic_universe`.

`SFA.accepts(...)` uses set-of-states semantics. Operations such as `complement()`,
`difference()`, `union()`, `intersection()`, and `concretize()` determinize
internally when needed.

## Concrete Workflow Integration

`SFA.from_acceptor(dfa)` lifts an existing concrete `DFA`-style acceptor into an
`SFA` using singleton `SetPredicate` guards.

`SFA.to_regex()` reuses the existing concrete DFA regex pipeline by first calling
`concretize()` and then delegating to `DFA.to_regex()`.

## Semantic Requirements

- `Z3Predicate` formulas are interpreted under the automaton-wide symbolic
  universe instead of carrying per-guard universes.
- Symbolic Boolean operations between automata require the same symbolic sort and
  equivalent symbolic universes.

- `SetPredicate.negate()` requires a finite universe.
- `SFA.complete()` and `SFA.complement()` require either a finite `alphabet` or a
  `predicate_factory` capable of producing a bottom predicate over the intended
  symbolic universe.
- `SFA.save()` and `SFA.load()` are finite-alphabet workflows. They serialize
  concrete symbols, not symbolic formulas.
- `SFA.concretize()` requires a finite alphabet and intentionally flattens guards
  to explicit transitions.

```bash
python mini2fst.py mini.l
```

```bash
python url2fst.py url.l
```
