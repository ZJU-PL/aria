# Program Synthesis

SyGuS and programming-by-example utilities for automatic program synthesis.

## Components

### PBE (Programming by Example)
- `pbe/task.py`: Typed task modeling and example validation
- `pbe/grammar.py`: Internal typed grammar for DSL construction
- `pbe/pbe_solver.py`: Main PBE solver
- `pbe/vsa.py`: Version Space Algebra implementation
- `pbe/expressions.py`: Expression types
- `pbe/expression_generators.py`: Typed candidate generation
- `pbe/expression_to_smt.py`: SMT conversion utilities
- `pbe/smt_verifier.py`: SMT-based verification
- `pbe/smt_pbe_solver.py`: PBE solver with SMT-backed validation

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
from aria.synthesis.pbe import PBESolver, PBETask

examples = [
    {"flag": True, "x": 10, "output": 10},
    {"flag": False, "x": 10, "output": 0},
    {"flag": True, "x": 3, "output": 3},
]

task = PBETask.from_examples(examples)
solver = PBESolver(max_expression_depth=2)
result = solver.synthesize(task.as_examples())

print(task.statistics())
print(result.expression)
```

```python
ambiguous_examples = [
    {"x": 1, "output": 1},
    {"x": 2, "output": 2},
]

refined = solver.synthesize_with_oracle(
    ambiguous_examples,
    lambda assignment: abs(assignment["x"]),
)

print(refined.expression)  # abs(x)
print(refined.statistics["ambiguity_resolved"])
```

## PBE Highlights

- Typed task inference tracks theory, input sorts, and output sort explicitly.
- An internal typed grammar drives enumeration across theories.
- Mixed-input tasks are supported, such as string synthesis with integer indices.
- Boolean inputs participate directly in conditional synthesis.
- Oracle-guided refinement can resolve underconstrained specifications.
- Ambiguous solutions expose distinguishing inputs and ranked alternatives.
