# Program Synthesis

SyGuS solvers for automatic program synthesis.

## Components

### PBE (Programming by Example)
- `pbe/pbe_solver.py`: Main PBE solver
- `pbe/vsa.py`: Version Space Algebra implementation
- `pbe/expressions.py`: Expression types
- `pbe/expression_generators.py`: Generator for candidate expressions
- `pbe/expression_to_smt.py`: SMT conversion utilities
- `pbe/smt_verifier.py`: SMT-based verification
- `pbe/smt_pbe_solver.py`: Enhanced PBE with SMT

### CVC5 Integration
- `cvc5/sygus_inv.py`: SyGuS invariant synthesis
- `cvc5/sygus_pbe.py`: SyGuS PBE for strings

### Spyro
- `spyro/spyro.py`: Main entry point
- `spyro/spyro_parser.py`: Template parser
- `spyro/property_synthesizer.py`: Property synthesis
- `spyro/input_generator.py`: Input generation

## What is SyGuS?

SyGuS (Syntax-Guided Synthesis) asks: given a specification (usually a set of input-output examples or a logical constraint), find a program that satisfies the specification.

## Usage

```python
from aria.synthesis.pbe import PBE solver

solver = PBESolver()
result = solver.synthesize(specification)
```
