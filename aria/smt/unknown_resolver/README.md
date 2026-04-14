# Unknown Result Resolver

This package contains a mutation-based helper for formulas where an external SMT
solver returns `unknown`.

## Main Modules

- `resolver.py`: `SAEUnknownResolver`, which explores structural and
  predicate-strengthening/weakening mutations in an attempt to turn `unknown`
  into a definitive `sat` or `unsat` answer.
- `main.py`: command-line entry point for running the resolver on an SMT-LIB2
  file against a chosen external solver binary.

## Approach

`SAEUnknownResolver` works in several stages:

1. convert the input formula to CNF-like structure with Z3's `tseitin-cnf`
   tactic;
2. try single mutations, such as:
   - dropping conjuncts or disjuncts;
   - instantiating a small number of variables with simple constants;
   - predicate-strengthening / predicate-weakening transformations for
     arithmetic literals;
3. try a small number of mutation combinations;
4. run a bounded-depth search over simple mutation sequences.

Each candidate formula is serialized back to SMT-LIB and checked by the chosen
external solver process.

## CLI Usage

```bash
python -m aria.smt.unknown_resolver.main \
  path/to/formula.smt2 \
  /path/to/solver \
  --timeout 30 \
  --max-depth 3 \
  --max-mutations 20
```

## Notes

- This package is heuristic and best viewed as an experiment aid, not a
  completeness-preserving decision procedure.
- The resolver only reports a decisive result when one of the mutated formulas
  yields `sat` or `unsat`.
- The external solver path must point to an executable that accepts an SMT-LIB2
  file path and prints a standard SMT result.
