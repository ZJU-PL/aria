# Automata

# Symbolic Automata

https://github.com/spencerwuwu/symautomata

A python framework for working with Automata. The framework contains a python implementation and C bindings using pywrapfst (optional).
This framework is part of the [lightbulb-framework](https://github.com/lightbulb-framework/lightbulb-framework).

## SFA API

The symbolic finite automata entrypoint is `aria.automata.symautomata.SFA`.

- `Predicate` is the abstract guard contract.
- `SetPredicate` is the finite-alphabet guard implementation.
- `Z3Predicate` is the solver-backed guard implementation for symbolic alphabets.

`SFA.accepts(...)` uses set-of-states semantics. Operations such as `complement()`,
`difference()`, `union()`, `intersection()`, and `concretize()` determinize
internally when needed.

## Concrete Workflow Integration

`SFA.from_acceptor(dfa)` lifts an existing concrete `DFA`-style acceptor into an
`SFA` using singleton `SetPredicate` guards.

`SFA.to_regex()` reuses the existing concrete DFA regex pipeline by first calling
`concretize()` and then delegating to `DFA.to_regex()`.

## Semantic Requirements

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
