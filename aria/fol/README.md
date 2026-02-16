# First-Order Logic

Educational theorem proving and reasoning tools for first-order logic.

## Components

- `miniprover/`: An educational automated theorem prover

## Miniprover

An complete proof system based on sequent calculus. For any provable formula,
this prover is guaranteed to find the proof (though may not terminate for unprovable formulas).

Features:
- Proof steps shown as sequents
- Command-line interface with parser
- Support for: variables, functions, predicates, connectives, quantifiers
- Axiom and lemma management

## Usage

```bash
# Run the theorem prover
python -m aria.fol.miniprover.main
```

For detailed usage, see `miniprover/README.md`.
